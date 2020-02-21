from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from newstrends import azure, utils
from newstrends.data import mysql


def search_keyword(keyword):
    field = ['title', 'date', 'publisher']
    date_from = datetime.now() - timedelta(days=7)
    entities = mysql.select_articles(field, date_from=date_from)
    titles = utils.preprocess([e[0] for e in entities])
    others = [(e[1].replace(hour=0, minute=0, second=0), e[2]) for e in entities]

    searched = []
    for title, entry in zip(titles, others):
        if title.find(keyword) >= 0:
            searched.append((title, *entry))
    return pd.DataFrame(searched, columns=field)


def main():
    keyword = '코로나'
    sentiments = ['positive', 'neutral', 'negative']
    df = search_keyword(keyword)
    scores = azure.compute_scores(df['title'])
    df['pos_score'] = scores[:, 0]
    df['neu_score'] = scores[:, 1]
    df['neg_score'] = scores[:, 2]
    df['polarity'] = np.maximum(scores[:, 0], scores[:, 2])

    for pub, df_ in df.groupby(by='publisher'):
        avg_score = df_[['pos_score', 'neu_score', 'neg_score']].mean(axis=0)
        print(f'Publisher: {pub}')
        print('Overall sentiment: {} ({})'.format(
            sentiments[np.argmax(avg_score)],
            ', '.join(f'{e:.3f}' for e in avg_score)))

        print('Articles:')
        df_ = df_.sort_values(by='date', ascending=False)
        for date, df__ in df_.groupby(by='date', sort=False):
            row = df__.sort_values(by='polarity', ascending=False).iloc[0, :]
            score = row[['pos_score', 'neu_score', 'neg_score']]
            print('[{}] ({}) {}'.format(
                row['date'].strftime('%Y/%m/%d'),
                ', '.join(f'{e:.3f}' for e in score),
                row['title']))
        print()


if __name__ == '__main__':
    main()
