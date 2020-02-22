import io
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch import nn, optim

from newstrends import azure
from newstrends.data import mysql


def preprocess(articles):
    stopwords = ['&#039;', '&quot;', '<span>', '</span>', '<span id="divTitle">',
                 '</font>', '<b>', '</b>']
    replace_dict = {'···': '…', '...': '…', '..': '…'}
    new_articles = []
    for article in articles:
        for word in stopwords:
            article = article.replace(word, '')
        for word, replace in replace_dict.items():
            article = article.replace(word, replace)
        article = re.sub(' +', ' ', article)
        if article.startswith(' '):
            article = article[1:]
        new_articles.append(article)
    return new_articles


def train_sentencepiece(title_path, model_path,
                        vocab_size=2048,
                        model_type='unigram',
                        character_coverage=0.9995):
    assert model_type in {'unigram', 'bpe', 'char', 'word'}
    model_prefix = model_path[:model_path.rfind('.')]
    spm.SentencePieceTrainer.Train(
        f'--input={title_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--character_coverage={character_coverage}')


def load_sentencepiece(path):
    model_path = os.path.join(path, 'spm.model')
    if not os.path.exists(path):
        title_path = os.path.join(path, 'titles.txt')
        titles = mysql.select_all_titles(preprocess=True)
        titles = np.array(titles, dtype=str)
        os.makedirs(os.path.dirname(title_path), exist_ok=True)
        np.savetxt(title_path, titles, fmt='%s')
        train_sentencepiece(title_path, model_path)
    model = spm.SentencePieceProcessor()
    model.Load(model_path)
    return model


def read_vocabulary(path):
    count = 3
    vocabulary = []
    with open(os.path.join(path, 'spm.vocab')) as f:
        for line in f:
            if count > 0:
                count -= 1
                continue
            words = line.strip().split('\t')
            vocabulary.append(words[0])
    return vocabulary


def to_integer_matrix(pieces, vocabulary=None, padding='first'):
    assert padding in {'first', 'last'}

    if vocabulary is None:
        vocabulary = list(set(p for pp in pieces for p in pp))
    piece_dict = {p: i for (i, p) in enumerate(vocabulary)}

    indices = []
    for i, pp in enumerate(pieces):
        indices.append([piece_dict[p] for p in pp if p in piece_dict])
    max_len = max(len(index) for index in indices)

    num_data = len(pieces)
    matrix = np.full((num_data, max_len), -1, dtype=np.int64)
    for i, index in enumerate(indices):
        k = 0
        if padding == 'first':
            k = max_len - len(index)
        for j, d in enumerate(index):
            matrix[i, j + k] = d
    return torch.from_numpy(matrix)


def to_multi_hot_matrix(pieces, vocabulary=None):
    if vocabulary is None:
        vocabulary = list(set(p for pp in pieces for p in pp))
    piece_dict = {p: i for (i, p) in enumerate(vocabulary)}
    num_data = len(pieces)
    vocab_size = len(vocabulary)
    matrix = np.zeros((num_data, vocab_size), dtype=np.float32)
    for i, pp in enumerate(pieces):
        for p in pp:
            if p in piece_dict:
                matrix[i, piece_dict[p]] = 1
    return torch.from_numpy(matrix)


def train_model(model, loader, num_epochs=1000, lr=1e-3, print_every=1,
                patience=0):
    loss_func = nn.CrossEntropyLoss()
    opt_module = optim.Adam(model.parameters(), lr)
    device = to_device()

    best_loss = np.inf
    best_epoch = 0
    saved_model = None

    for epoch in range(num_epochs):
        loss_sum, num_data = 0, 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            opt_module.step()
            loss_sum += loss.item() * x.size(0)
            num_data += x.size(0)

        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1:3d}: {loss_sum / num_data}')

        if loss_sum < best_loss:
            best_loss = loss_sum
            best_epoch = epoch
            saved_model = io.BytesIO()
            torch.save(model.state_dict(), saved_model)

        if patience > 0 and epoch >= best_epoch + patience:
            break

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))


def to_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def search_keywords_as_dataframe(*keywords, num_days='all', ignore_time=True):
    field = ['title', 'date', 'publisher']
    date_from = None
    if isinstance(num_days, int):
        date_from = datetime.now() - timedelta(days=num_days)
    entities = mysql.select_articles(field, date_from=date_from)
    titles = preprocess([e[0] for e in entities])
    others = [e[1:] for e in entities]

    searched = []
    for title, entry in zip(titles, others):
        found = False
        for keyword in keywords:
            if title.find(keyword) >= 0:
                found = True
                break
        if found:
            searched.append((title, *entry))
    df = pd.DataFrame(searched, columns=field)

    if ignore_time:
        df['date'] = df['date'].apply(lambda x: x.date())

    return df


def search_keyword_sentiment(keyword):
    keywords = [keyword, keyword.replace(' ', '')]
    df = search_keywords_as_dataframe(*keywords, num_days=7)
    if df.empty:
        return None
    scores = azure.compute_scores(df['title'])
    sentiments = np.array(['긍정적', '중립적', '부정적'], dtype=str)
    df['pos_score'] = scores[:, 0]
    df['neu_score'] = scores[:, 1]
    df['neg_score'] = scores[:, 2]
    df['sentiment'] = sentiments[scores.argmax(axis=1)]
    return df


def to_keywords(articles):
    stopwords = [
        '&#039;', '&quot;', '<span>', '</span>', '<span id="divTitle">', '<b>',
        '</font>', '</b>', '포토', '없는', '것인가', '[포토]', '속보', '[속보]', '첫',
        '등', '중', '수', '외', '전', '내', '것', '만에', '더', '논란', '·', '발표',
        '안', '후', '출시', '위한', 'ET투자뉴스', '\u200b']

    parsed = []
    for title in articles:
        for w in [',', "','", '[', ']', '(', ')']:
            title = title.replace(w, ' ')
        for w in ["'", '"', "'", '‘', '’']:
            title = title.replace(w, '')
        parsed.append([word for word in title.split() if word not in stopwords])
    return parsed


def find_popular_keywords(num_words=20, num_days=3):
    entries = mysql.select_articles(
        field=['title', 'description', 'date'],
        date_from=datetime.now() - timedelta(num_days))
    words_dict = defaultdict(lambda: [])
    for e in entries:
        words_dict[e[2].date()].append(e[0])

    data = []
    for date in sorted(words_dict.keys()):
        keywords = to_keywords(words_dict[date])
        keywords = [w for words in keywords for w in words]
        keywords = Counter(keywords).most_common(num_words)
        for word, count in keywords:
            data.append((date, word, count))
    return pd.DataFrame(data, columns=['date', 'word', 'count'])


def compute_sentence_polarity(sentence):
    return dict(보수=0.6, 진보=0.4)
