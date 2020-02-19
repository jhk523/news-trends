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


def to_multi_hot_matrix(pieces, vocabulary=None):
    if vocabulary is None:
        vocabulary = list(set(p for pp in pieces for p in pp))
    piece_dict = {p: i for (i, p) in enumerate(vocabulary)}
    num_data = len(pieces)
    num_pieces = len(vocabulary)
    matrix = np.zeros((num_data, num_pieces), dtype=np.float32)
    for i, pp in enumerate(pieces):
        for p in pp:
            if p in piece_dict:
                matrix[i, piece_dict[p]] = 1
    return torch.from_numpy(matrix)


def train_model(model, loader, num_epochs=1000, lr=1e-3, print_every=1):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        loss_sum = 0
        num_data = 0
        for x, y in loader:
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            num_data += x.size(0)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1:3d}: {loss_sum / num_data}')
