import os

import numpy as np

from newstrends import utils
from newstrends.data import mysql


def main():
    model_path = '../../data/sentencepiece'
    os.makedirs(model_path, exist_ok=True)

    title_path = os.path.join(model_path, 'titles.txt')
    titles = mysql.select_all_titles(preprocess=True)
    titles = np.array(titles, dtype=str)
    np.savetxt(title_path, titles, fmt='%s')

    model_path = os.path.join(model_path, 'spm.model')
    utils.train_sentencepiece(title_path, model_path)


if __name__ == '__main__':
    main()
