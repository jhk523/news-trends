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


def save_as_pieces(model, path):
    entries = mysql.select_articles(
        field=['title', 'publisher'], publishers=['조선일보', '경향신문', '한겨례'])
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


def print_predictions(model, loader, titles):
    count = 0
    for x, y in loader:
        y_pred = torch.softmax(model(x), dim=1)
        for i in range(x.size(0)):
            pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[i])
            label = '조선일보' if y[i] == 0 else '한겨레경향'
            print(f'Title: {titles[count]}')
            print(f'Prediction: ({pred_str})')
            print(f'True label: ({label})')
            print()
            count += 1


def start_interactive_session(model, spm_model, unique_pieces):
    device = utils.to_device()
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        pieces = spm_model.encode_as_pieces(sentence)
        matrix = utils.to_integer_matrix([pieces], unique_pieces).to(device)
        # matrix = utils.to_multi_hot_matrix([pieces], unique_pieces).to(device)
        y_pred = torch.softmax(model(matrix), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def main():
    out_path = '../../out'
    spm_model = utils.load_spm(path=f'{out_path}/spm/spm.model')
    save_as_pieces(spm_model, path=f'{out_path}/train')

    label_map = dict(조선일보=0, 경향신문=1, 한겨례=1)

    pieces = read_pieces(f'{out_path}/train/pieces.tsv')
    vocabulary = list(set(p for pp in pieces for p in pp))
    # features = utils.to_multi_hot_matrix(pieces, vocabulary)
    features = utils.to_integer_matrix(pieces, vocabulary)
    labels = read_labels_as_tensor(f'{out_path}/train/labels.tsv', label_map)

    vocab_size = len(vocabulary)
    num_classes = 2
    embedding_dim = 8
    batch_size = 256

    device = utils.to_device()
    # cls_model = models.SoftmaxClassifier(
    #     vocab_size, num_classes, embedding_dim).to(device)
    cls_model = models.RNNClassifier(
        vocab_size, num_classes, embedding_dim).to(device)
    cls_path = f'{out_path}/pub/model.pth'

    if not os.path.exists(cls_path):
        os.makedirs(os.path.dirname(cls_path), exist_ok=True)
        loader = DataLoader(
            TensorDataset(features, labels), batch_size, shuffle=True)
        utils.train_model(
            cls_model, loader, lr=1e-4, num_epochs=1500, print_every=100)
        torch.save(cls_model.state_dict(), cls_path)
    else:
        cls_model.load_state_dict(torch.load(cls_path))

    title_path = f'{out_path}/train/titles.tsv'
    titles = [e.strip() for e in open(title_path).readlines()]
    # print_predictions(cls_model, loader, titles)
    start_interactive_session(cls_model, spm_model, vocabulary)


if __name__ == '__main__':
    main()
