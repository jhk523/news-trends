import os

import numpy as np
import sentencepiece as spm

from newstrends import utils
from newstrends.data import mysql


def write_articles(articles, path):
    articles = utils.preprocess(articles)
    articles = np.array(articles, dtype=str)
    np.savetxt(path, articles, fmt='%s')


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
    articles = mysql.select_all_articles()
    titles = mysql.select_articles(
        field='title', publishers=['조선일보', '한겨례'])
    titles = utils.preprocess(titles)

    article_path = '../out/spm/articles.txt'
    os.makedirs(os.path.dirname(article_path), exist_ok=True)
    write_articles(articles, article_path)

    model_path = '../out/spm/test.model'
    if not os.path.exists(model_path):
        train_spm(article_path, model_path)
    model = spm.SentencePieceProcessor()
    model.Load(model_path)

    piece_path = '../out/spm/pieces.tsv'
    with open(piece_path, 'w') as f:
        for title in titles:
            pieces = model.EncodeAsPieces(title)
            f.write('\t'.join(pieces) + '\n')


if __name__ == '__main__':
    main()
