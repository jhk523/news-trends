import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from newstrends import utils, models
from scripts.publisher import start_interactive_session


def read_reviews():
    df = pd.read_csv('../data/ratings.txt', delimiter='\t')
    df = df[df['document'].notnull()]
    reviews = df['document'].tolist()
    labels = df['label'].tolist()
    return reviews, labels


def main():
    out_path = '../out'
    model_path = os.path.join(out_path, 'model/spm.model')
    spm_model = utils.load_spm(model_path)
    reviews, labels = read_reviews()

    piece_list = []
    for review in reviews:
        piece_list.append(spm_model.encode_as_pieces(review))

    vocab = utils.read_vocabulary('../out/model/spm.vocab')
    features = utils.to_integer_matrix(piece_list, vocab)
    labels = torch.tensor(labels)
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
    cls_model = models.GRUClassifier(vocab_size=len(vocab), num_classes=2)
    utils.train_model(cls_model, loader, lr=1e-4, print_every=1, num_epochs=100)

    start_interactive_session(cls_model, spm_model, vocab)


if __name__ == '__main__':
    main()
