import os
import typing

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from newstrends import utils
from newstrends.data import mysql


class EmbeddingModel(nn.Module):
    def __init__(self, num_pieces, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_pieces, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 2)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out = torch.matmul(x, self.embedding.weight)
        return self.linear(out)


def save_strings(path, name, data):
    if isinstance(data, list):
        data = np.array(data, dtype=str)
    np.savetxt(os.path.join(path, name), data, fmt='%s')


def save_as_pieces(model, path):
    entries = mysql.select_articles(
        field=['title', 'publisher'], publishers=['조선일보', '한겨례'])
    titles = [e[0] for e in entries]
    titles = utils.preprocess(titles)
    publishers = [e[1] for e in entries]

    os.makedirs(path, exist_ok=True)
    save_strings(path, 'titles.tsv', titles)
    save_strings(path, 'labels.tsv', publishers)

    piece_list = []
    piece_path = os.path.join(path, 'pieces.tsv')
    with open(piece_path, 'w') as f1:
        for title in titles:
            pieces = model.EncodeAsPieces(title)
            piece_list.append(pieces)
            f1.write('\t'.join(pieces) + '\n')


def to_multi_hot(pieces, unique_pieces=None):
    if unique_pieces is None:
        unique_pieces = list(set(p for pp in pieces for p in pp))
    piece_dict = {p: i for (i, p) in enumerate(unique_pieces)}
    num_data = len(pieces)
    num_pieces = len(unique_pieces)
    matrix = np.zeros((num_data, num_pieces), dtype=np.float32)
    for i, pp in enumerate(pieces):
        for p in pp:
            matrix[i, piece_dict[p]] = 1
    return matrix


def read_pieces(path):
    pieces = []
    with open(path) as f:
        for line in f:
            pieces.append(line.strip().split('\t'))
    return pieces


def read_labels_as_tensor(path):
    labels = []
    label_map = {}
    with open(path) as f:
        for line in f:
            label = line.strip()
            if label not in label_map:
                label_map[label] = len(label_map)
            labels.append(label_map[label])
    return torch.tensor(labels)


def print_predictions(model, loader, titles):
    count = 0
    for x, y in loader:
        y_pred = torch.softmax(model(x), dim=1)
        for i in range(x.size(0)):
            pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[i])
            label = '조선일보' if y[i] == 0 else '한겨레'
            print(f'Title: {titles[count]}')
            print(f'Prediction: ({pred_str})')
            print(f'True label: ({label})')
            print()
            count += 1


def start_interactive_session(model, spm_model, unique_pieces):
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        pieces = spm_model.encode_as_pieces(sentence)
        matrix = torch.from_numpy(to_multi_hot([pieces], unique_pieces))
        y_pred = torch.softmax(model(matrix), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def train_model(model, loader):
    num_epochs = 1000
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1:3d}: {loss_sum / num_data}')


def main():
    spm_model = utils.load_spm(path='../out/spm/model/spm.model')
    save_as_pieces(spm_model, path='../out/spm/train')

    pieces = read_pieces('../out/spm/train/pieces.tsv')
    vocabulary = list(set(p for pp in pieces for p in pp))
    features = torch.from_numpy(to_multi_hot(pieces, vocabulary))
    labels = read_labels_as_tensor('../out/spm/train/labels.tsv')

    cls_model = EmbeddingModel(num_pieces=len(vocabulary))
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
    train_model(cls_model, loader)

    title_path = '../out/spm/train/titles.tsv'
    titles = [e.strip() for e in open(title_path).readlines()]
    print_predictions(cls_model, loader, titles)
    start_interactive_session(cls_model, spm_model, vocabulary)


if __name__ == '__main__':
    main()
