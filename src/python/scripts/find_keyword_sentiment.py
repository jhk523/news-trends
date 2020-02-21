import numpy as np

from newstrends import utils


def main():
    keyword = '사랑의 불시착'
    df = utils.search_keyword_sentiment(keyword)
    sentiments = ['긍정적', '중립적', '부정적']

    if df is None:
        return

    for pub, df_ in df.groupby(by='publisher'):
        print(f'[Publisher] {pub}')

        df_ = df_.sort_values(by='date', ascending=False)
        for date, df__ in df_.groupby(by='date', sort=False):
            avg_score = df__[['pos_score', 'neu_score', 'neg_score']].values.mean(axis=0)
            print('[{}] {:2d}개 기사 - {} ({})'.format(
                date.strftime('%Y/%m/%d'),
                df__.shape[0],
                sentiments[np.argmax(avg_score)],
                ', '.join(f'{e:.3f}' for e in avg_score)))
        print()


if __name__ == '__main__':
    main()
