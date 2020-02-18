import os

import numpy as np
import sentencepiece as spm

from newstrends import utils
from newstrends.data import mysql


def read_titles(path):
    with open(path) as f:
        return f.readlines()


def write_titles(path):
    titles = [e[0] for e in mysql.select_news_articles(field='title')]
    titles = utils.preprocess(titles)
    titles = np.array(titles, dtype=str)
    np.savetxt(path, titles, fmt='%s')


def train_spm(title_path, model_path):
    model_prefix = model_path[:model_path.rfind('.')]
    vocab_size = 8000
    model_type = 'unigram'  # unigram (default), bpe, char, or word.
    spm.SentencePieceTrainer.Train(
        f'--input={title_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type}')


def main():
    title_path = '../out/spm/titles.txt'
    os.makedirs(os.path.dirname(title_path), exist_ok=True)
    write_titles(title_path)

    model_path = '../out/spm/test.model'
    if not os.path.exists(model_path):
        train_spm(title_path, model_path)
    model = spm.SentencePieceProcessor()
    model.Load(model_path)

    piece_list = []
    titles = read_titles(title_path)
    for title in titles:
        piece_list.append(model.EncodeAsPieces(title))

    piece_path = '../out/spm/pieces.tsv'
    with open(piece_path, 'w') as f:
        for pieces in piece_list:
            f.write('\t'.join(pieces) + '\n')


if __name__ == '__main__':
    main()
