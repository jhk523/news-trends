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


def main():
    out_path = '../out'
    model_path = os.path.join(out_path, 'model/spm.model')
    model = utils.load_spm(model_path)
    reviews, labels = read_reviews()

    piece_list = []
    for review in reviews:
        piece_list.append(model.encode_as_pieces(review))

    vocab = utils.read_vocabulary('../out/model/spm.vocab')
    features = utils.to_multi_hot_matrix(piece_list, vocab)
    labels = torch.tensor(labels)
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
    model = models.EmbeddingModel(num_pieces=len(vocab), num_classes=2)
    utils.train_model(model, loader, lr=1e-4, print_every=1)


if __name__ == '__main__':
    main()
