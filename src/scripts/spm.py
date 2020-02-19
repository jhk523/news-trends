import os

import numpy as np
import sentencepiece as spm

from newstrends import utils
from newstrends.data import mysql


def train_spm(title_path, model_path,
              vocab_size=2048,
              model_type='unigram',
              character_coverage=0.9995):
    # model_type is in { unigram, bpe, char, word }.
    model_prefix = model_path[:model_path.rfind('.')]
    spm.SentencePieceTrainer.Train(
        f'--input={title_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--character_coverage={character_coverage}')


def write_articles(articles, path):
    articles = utils.preprocess(articles)
    articles = np.array(articles, dtype=str)
    np.savetxt(path, articles, fmt='%s')


def main():
    out_path = '../out/model'
    os.makedirs(out_path, exist_ok=True)

    title_path = os.path.join(out_path, 'titles.txt')
    all_titles = mysql.select_all_titles()
    write_articles(all_titles, title_path)

    model_path = os.path.join(out_path, 'spm.model')
    train_spm(title_path, model_path)


if __name__ == '__main__':
    main()
