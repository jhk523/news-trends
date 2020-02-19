import typing

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from newstrends.spm import load_spm


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


def print_predictions(model, loader):
    title_path = '../out/spm/titles.tsv'
    titles = [e.strip() for e in open(title_path).readlines()]

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


def start_interactive_session(model, unique_pieces):
    spm_model = load_spm(path='../out/spm/model/spm.model')
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        pieces = spm_model.encode_as_pieces(sentence)
        matrix = torch.from_numpy(to_multi_hot([pieces], unique_pieces))
        y_pred = torch.softmax(model(matrix), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def main():
    piece_path = '../out/spm/pieces.tsv'
    pieces = []
    with open(piece_path) as f:
        for line in f:
            pieces.append(line.strip().split('\t'))
    unique_pieces = list(set(p for pp in pieces for p in pp))
    features = torch.from_numpy(to_multi_hot(pieces, unique_pieces))

    label_path = '../out/spm/labels.tsv'
    labels = []
    with open(label_path) as f:
        for line in f:
            if line.strip() == '조선일보':
                labels.append(0)
            elif line.strip() == '한겨례':
                labels.append(1)
            else:
                raise ValueError()
    labels = torch.tensor(labels)

    unique_pieces = list(set(p for pp in pieces for p in pp))
    num_pieces = len(unique_pieces)
    model = EmbeddingModel(num_pieces)

    num_data = len(pieces)
    batch_size = 256
    num_epochs = 1000

    dataset = TensorDataset(features, labels)
    data = DataLoader(dataset, batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        loss_sum = 0
        for x, y in data:
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1:3d}: {loss_sum / num_data}')

    print_predictions(model, data)
    start_interactive_session(model, unique_pieces)


if __name__ == '__main__':
    main()
