import numpy as np

from newstrends import azure
from newstrends.data import mysql


def print_scores(documents, scores):
    sentiments = ['positive', 'neutral', 'negative']
    for t, s in zip(documents, scores):
        scores_str = ', '.join(f'{e:.3f}' for e in s)
        print(f'Document: {t}')
        print(f'Sentiment: {sentiments[np.argmax(s)]}')
        print(f'Scores: ({scores_str})')
        print()


def main():
    titles = mysql.select_all_titles(preprocess=True)
    titles = titles[:1000]
    scores = azure.compute_scores(titles)
    print_scores(titles, scores)


if __name__ == '__main__':
    main()
