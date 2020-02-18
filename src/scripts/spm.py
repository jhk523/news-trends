import os

import numpy as np
import sentencepiece as spm

from newstrends import utils
from newstrends.data import mysql


def write_articles(articles, path):
    articles = utils.preprocess(articles)
    articles = np.array(articles, dtype=str)
    np.savetxt(path, articles, fmt='%s')


def train_spm(title_path, model_path, vocab_size=2048, model_type='unigram'):
    # model_type is in { unigram, bpe, char, word }.
    model_prefix = model_path[:model_path.rfind('.')]
    spm.SentencePieceTrainer.Train(
        f'--input={title_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type}')


def encode_as_pieces(model, out_path):
    entries = mysql.select_articles(
        field=['title', 'publisher'], publishers=['조선일보', '한겨례'])
    titles = [e[0] for e in entries]
    titles = utils.preprocess(titles)
    publishers = [e[1] for e in entries]

    title_path = os.path.join(out_path, 'titles.tsv')
    piece_path = os.path.join(out_path, 'pieces.tsv')
    label_path = os.path.join(out_path, 'labels.tsv')
    with open(piece_path, 'w') as f1:
        for title in titles:
            pieces = model.EncodeAsPieces(title)
            f1.write('\t'.join(pieces) + '\n')

    np.savetxt(title_path, np.array(titles, dtype=str), fmt='%s')
    np.savetxt(label_path, np.array(publishers, dtype=str), fmt='%s')


def main():
    out_path = '../out/spm'
    os.makedirs(os.path.join(out_path, 'model'), exist_ok=True)

    title_path = os.path.join(out_path, 'model/titles.txt')
    all_titles = mysql.select_all_titles()
    write_articles(all_titles, title_path)

    model_path = os.path.join(out_path, 'model/spm.model')
    if not os.path.exists(model_path):
        train_spm(title_path, model_path)
    model = spm.SentencePieceProcessor()
    model.Load(model_path)
    encode_as_pieces(model, out_path)


if __name__ == '__main__':
    main()
