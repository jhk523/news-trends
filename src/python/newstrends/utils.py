import io
import re
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


def load_spm(path):
    model = spm.SentencePieceProcessor()
    model.Load(path)
    return model


def read_vocabulary(path):
    count = 3
    vocabulary = []
    with open(path) as f:
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


def search_keyword_as_dataframe(keyword, num_days='all', ignore_time=True):
    field = ['title', 'date', 'publisher']
    date_from = None
    if isinstance(num_days, int):
        date_from = datetime.now() - timedelta(days=num_days)
    entities = mysql.select_articles(field, date_from=date_from)
    titles = preprocess([e[0] for e in entities])
    others = [e[1:] for e in entities]

    searched = []
    for title, entry in zip(titles, others):
        if title.find(keyword) >= 0:
            searched.append((title, *entry))
    df = pd.DataFrame(searched, columns=field)

    if ignore_time:
        df['date'] = df['date'].apply(lambda x: x.replace(hour=0, minute=0, second=0))
    return df


def search_keyword_sentiment(keyword):
    df = search_keyword_as_dataframe(keyword, num_days=7)
    scores = azure.compute_scores(df['title'])
    df['pos_score'] = scores[:, 0]
    df['neu_score'] = scores[:, 1]
    df['neg_score'] = scores[:, 2]
    return df
