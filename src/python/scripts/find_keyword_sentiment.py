import numpy as np

from newstrends import azure, utils


def main():
    keyword = '코로나'
    sentiments = ['positive', 'neutral', 'negative']
    df = utils.search_keyword(keyword, num_days=7)
    scores = azure.compute_scores(df['title'])
    df['pos_score'] = scores[:, 0]
    df['neu_score'] = scores[:, 1]
    df['neg_score'] = scores[:, 2]
    df['polarity'] = np.maximum(scores[:, 0], scores[:, 2])

    for pub, df_ in df.groupby(by='publisher'):
        avg_score = df_[['pos_score', 'neu_score', 'neg_score']].values.mean(axis=0)
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
