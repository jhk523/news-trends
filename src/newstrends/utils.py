import re

import numpy as np
import sentencepiece as spm
import torch
from torch import nn, optim


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


def to_integer_matrix(pieces, vocabulary=None):
    if vocabulary is None:
        vocabulary = list(set(p for pp in pieces for p in pp))
    piece_dict = {p: i for (i, p) in enumerate(vocabulary)}

    indices = []
    max_len = 0
    for i, pp in enumerate(pieces):
        index = []
        for p in pp:
            if p in piece_dict:
                index.append(piece_dict[p])
        if len(index) > max_len:
            max_len = len(index)
        indices.append(index)

    num_data = len(pieces)
    matrix = np.full((num_data, max_len), -1, dtype=np.int64)
    for i, dd in enumerate(indices):
        j = max_len - len(dd)
        for d in dd:
            matrix[i, j] = d
            j += 1

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


def train_model(model, loader, num_epochs=1000, lr=1e-3, print_every=1):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    device = to_device()

    for epoch in range(num_epochs):
        loss_sum, num_data = 0, 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            num_data += x.size(0)

        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1:3d}: {loss_sum / num_data}')


def to_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
