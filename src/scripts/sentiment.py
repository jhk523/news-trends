import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from newstrends import utils, models


def read_reviews():
    df = pd.read_csv('../data/ratings.txt', delimiter='\t')
    df = df[df['document'].notnull()]
    reviews = df['document'].tolist()
    labels = df['label'].tolist()
    return reviews, labels


def train_classifier(cls_model, spm_model, vocab):
    reviews, labels = read_reviews()
    pieces = [spm_model.encode_as_pieces(r) for r in reviews]
    features = utils.to_integer_matrix(pieces, vocab)
    labels = torch.tensor(labels)
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=512, shuffle=True)
    utils.train_model(
        cls_model, loader, lr=1e-4, print_every=5, num_epochs=100)


def start_interactive_session(cls_model, spm_model, vocabulary):
    device = utils.to_device()
    while True:
        print('Sentence:', end=' ')
        sentence = input()
        pieces = spm_model.encode_as_pieces(sentence)
        features = utils.to_integer_matrix([pieces], vocabulary).to(device)
        y_pred = torch.softmax(cls_model(features), dim=1)

        pred_str = ', '.join(f'{e * 100:.1f}' for e in y_pred[0])
        print(f'Prediction: ({pred_str})')
        print()


def main():
    out_path = '../out'
    spm_path = os.path.join(out_path, 'spm')
    spm_model = utils.load_spm(os.path.join(spm_path, 'spm.model'))
    vocab = utils.read_vocabulary(os.path.join(spm_path, 'spm.vocab'))

    cls_path = os.path.join(out_path, 'sent/model.pth')
    cls_model = models.GRUClassifier(
        vocab_size=len(vocab), num_classes=2, embedding_dim=8, hidden_size=8) \
        .to(utils.to_device())
    if not os.path.exists(cls_path):
        train_classifier(cls_model, spm_model, vocab)
        os.makedirs(os.path.dirname(cls_path), exist_ok=True)
        torch.save(cls_model.state_dict(), cls_path)
    else:
        cls_model.load_state_dict(torch.load(cls_path))

    start_interactive_session(cls_model, spm_model, vocab)


if __name__ == '__main__':
    main()
