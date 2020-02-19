import os

import numpy as np

from newstrends import utils
from newstrends.data import mysql
from newstrends.spm import train_spm, load_spm


def write_articles(articles, path):
    articles = utils.preprocess(articles)
    articles = np.array(articles, dtype=str)
    np.savetxt(path, articles, fmt='%s')


def save_strings(path, name, data):
    if isinstance(data, list):
        data = np.array(data, dtype=str)
    np.savetxt(os.path.join(path, name), data, fmt='%s')


def save_as_pieces(model, out_path):
    entries = mysql.select_articles(
        field=['title', 'publisher'], publishers=['조선일보', '한겨례'])
    titles = [e[0] for e in entries]
    titles = utils.preprocess(titles)
    publishers = [e[1] for e in entries]

    os.makedirs(out_path, exist_ok=True)
    save_strings(out_path, 'titles.tsv', titles)
    save_strings(out_path, 'labels.tsv', publishers)

    piece_list = []
    piece_path = os.path.join(out_path, 'pieces.tsv')
    with open(piece_path, 'w') as f1:
        for title in titles:
            pieces = model.EncodeAsPieces(title)
            piece_list.append(pieces)
            f1.write('\t'.join(pieces) + '\n')


def main():
    out_path = '../out'
    os.makedirs(os.path.join(out_path, 'model'), exist_ok=True)

    title_path = os.path.join(out_path, 'model/titles.txt')
    all_titles = mysql.select_all_titles()
    write_articles(all_titles, title_path)

    model_path = os.path.join(out_path, 'model/spm.model')
    if not os.path.exists(model_path):
        train_spm(title_path, model_path)
    model = load_spm(model_path)
    save_as_pieces(model, os.path.join(out_path, 'train'))


if __name__ == '__main__':
    main()
