import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from newstrends import utils, models
from newstrends.data import mysql


def read_reviews(path):
    df = pd.read_csv(path, delimiter='\t')
    df = df[df['document'].notnull()]
    reviews = utils.preprocess(df['document'].tolist())
    labels = df['label'].tolist()
    return reviews, labels


def train_classifier(cls_model, spm_model, vocab):
    reviews, labels = read_reviews(path='../../data/ratings.txt')
    pieces = [spm_model.encode_as_pieces(r) for r in reviews]
    features = utils.to_integer_matrix(pieces, vocab, padding='first')
    # features = utils.to_multi_hot_matrix(pieces, vocab)
    dataset = TensorDataset(features, torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    utils.train_model(
        cls_model, loader, lr=1e-4, print_every=1, num_epochs=100)


def start_interactive_session(cls_model, spm_model, vocabulary):
    device = utils.to_device()
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        pieces = spm_model.encode_as_pieces(sentence)
        features = utils.to_integer_matrix([pieces], vocabulary, padding='first')
        # features = utils.to_multi_hot_matrix([pieces], vocabulary)
        y_pred = torch.softmax(cls_model(features.to(device)), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def test_for_article_titles(cls_model, spm_model, vocabulary):
    device = utils.to_device()
    titles = mysql.select_all_titles(preprocess=True)
    pieces = [spm_model.encode_as_pieces(t) for t in titles]
    features = utils.to_integer_matrix(pieces, vocabulary, padding='first')
    loader = DataLoader(TensorDataset(features), batch_size=1024)

    count = 0
    for x, in loader:
        y_pred = torch.softmax(cls_model(x.to(device)), dim=1)
        for i in range(x.size(0)):
            pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[i])
            print(f'Title: {titles[count]}')
            print(f'Prediction: ({pred_str})')
            print()
            count += 1


def main():
    out_path = '../../out'
    spm_path = os.path.join(out_path, 'spm')
    spm_model = utils.load_spm(os.path.join(spm_path, 'spm.model'))
    vocab = utils.read_vocabulary(os.path.join(spm_path, 'spm.vocab'))
    vocab_size = len(vocab)

    num_classes = 2
    embedding_dim = 64
    cell_type = 'lstm'
    num_layers = 1

    cls_path = os.path.join(out_path, 'sent/model.pth')
    # cls_model = models.TransformerClassifier(
    #     vocab_size=vocab_size, num_classes=1, embedding_dim=8) \
    #     .to(utils.to_device())
    cls_model = models.RNNClassifier(
        vocab_size, num_classes, embedding_dim, cell_type, num_layers) \
        .to(utils.to_device())
    # cls_model = models.SoftmaxClassifier(
    #     vocab_size=vocab_size, num_classes=2, embedding_dim=64) \
    #     .to(utils.to_device())

    if not os.path.exists(cls_path):
        train_classifier(cls_model, spm_model, vocab)
        os.makedirs(os.path.dirname(cls_path), exist_ok=True)
        torch.save(cls_model.state_dict(), cls_path)
    else:
        cls_model.load_state_dict(torch.load(cls_path))

    # start_interactive_session(cls_model, spm_model, vocab)
    test_for_article_titles(cls_model, spm_model, vocab)


if __name__ == '__main__':
    main()
