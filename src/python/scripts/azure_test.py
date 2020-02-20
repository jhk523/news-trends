import numpy as np

from newstrends import azure
from newstrends.data import mysql


def main():
    titles = mysql.select_all_titles(preprocess=True)
    titles = titles[:1000]
    scores = azure.compute_scores(titles)
    sentiments = ['positive', 'neutral', 'negative']

    for t, s in zip(titles, scores):
        scores_str = ', '.join(f'{e:.3f}' for e in s)
        print(f'Sentence: {t}')
        print(f'Overall sentiment: {sentiments[np.argmax(s)]}')
        print(f'Scores: ({scores_str})')
        print()


if __name__ == '__main__':
    main()
