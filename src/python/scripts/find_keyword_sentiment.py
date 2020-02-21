import numpy as np

from newstrends import utils


def main():
    keyword = '코로나'
    df = utils.search_keyword_sentiment(keyword)
    sentiments = ['긍정적', '중립적', '부정적']

    for pub, df_ in df.groupby(by='publisher'):
        print(f'[Publisher] {pub}')
        # print('Overall sentiment: {} ({})'.format(
        #     sentiments[np.argmax(avg_score)],
        #     ', '.join(f'{e:.3f}' for e in avg_score)))

        df_ = df_.sort_values(by='date', ascending=False)
        for date, df__ in df_.groupby(by='date', sort=False):
            avg_score = df__[['pos_score', 'neu_score', 'neg_score']].values.mean(axis=0)
            print('[{}] {} ({})'.format(
                date.strftime('%Y/%m/%d'),
                sentiments[np.argmax(avg_score)],
                ', '.join(f'{e:.3f}' for e in avg_score)))
            # row = df__.sort_values(by='polarity', ascending=False).iloc[0, :]
            # score = row[['pos_score', 'neu_score', 'neg_score']]
        print()


if __name__ == '__main__':
    main()
