import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from newstrends import utils, models
from newstrends.data import mysql


def save_strings(path, name, data):
    if isinstance(data, list):
        data = np.array(data, dtype=str)
    np.savetxt(os.path.join(path, name), data, fmt='%s')


def save_as_pieces(model, path, publishers):
    entries = mysql.select_articles(
        field=['title', 'publisher'], publishers=publishers)
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


def read_pieces(path):
    pieces = []
    with open(path) as f:
        for line in f:
            pieces.append(line.strip().split('\t'))
    return pieces


def read_labels_as_tensor(path, label_map):
    labels = []
    with open(path) as f:
        for line in f:
            labels.append(label_map[line.strip()])
    return torch.tensor(labels)


def print_predictions(model, loader, titles, path):
    model.eval()
    device = utils.to_device()
    count = 0
    f = open(path, 'w')
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = torch.softmax(model(x), dim=1)
        for i in range(x.size(0)):
            pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[i])
            label = '조선일보' if y[i] == 0 else '한겨레경향'
            f.write(f'Title: {titles[count]}\n')
            f.write(f'Prediction: ({pred_str})\n')
            f.write(f'True label: ({label})\n')
            f.write('\n')
            count += 1
    f.close()


def start_interactive_session(model, spm_model, unique_pieces):
    model.eval()
    device = utils.to_device()
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        if len(sentence.strip()) == 0:
            continue
        pieces = spm_model.encode_as_pieces(sentence)
        matrix = utils.to_integer_matrix([pieces], unique_pieces).to(device)
        y_pred = torch.softmax(model(matrix), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def main():
    publishers = ['조선일보', '경향신문', '한겨례']
    label_map = dict(조선일보=0, 경향신문=1, 한겨례=1)
    spm_path = '../../data/sentencepiece'
    pub_path = '../../data/publishers'
    num_classes = 2
    embedding_dim = 12
    batch_size = 512
    dropout = 0.5

    spm_model = utils.load_sentencepiece(spm_path)
    spm_vocab = utils.read_vocabulary(spm_path)
    vocab_size = len(spm_vocab)

    device = utils.to_device()
    cls_model = models.RNNClassifier(
        vocab_size, num_classes, embedding_dim, dropout=dropout).to(device)
    cls_path = os.path.join(pub_path, 'model.pth')

    if not os.path.exists(pub_path):
        train_path = os.path.join(pub_path, 'train')
        save_as_pieces(spm_model, train_path, publishers)

        pieces = read_pieces(os.path.join(train_path, 'pieces.tsv'))
        features = utils.to_integer_matrix(pieces, spm_vocab)
        labels = read_labels_as_tensor(
            os.path.join(train_path, 'labels.tsv'), label_map)
        loader = DataLoader(
            TensorDataset(features, labels), batch_size, shuffle=True)

        os.makedirs(os.path.dirname(cls_path), exist_ok=True)
        utils.train_model(
            cls_model, loader, lr=1e-4, num_epochs=1500, print_every=100, patience=100)
        torch.save(cls_model.state_dict(), cls_path)

        titles = open(os.path.join(train_path, 'titles.tsv')).readlines()
        titles = [e.strip() for e in titles]
        print_predictions(cls_model, loader, titles, f'{pub_path}/predictions.txt')
    else:
        cls_model.load_state_dict(torch.load(cls_path, map_location=device))

    start_interactive_session(cls_model, spm_model, spm_vocab)


if __name__ == '__main__':
    main()
