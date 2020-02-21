from newstrends import utils


def main():
    keyword = '코로나'
    df = utils.search_keyword(keyword, num_days=7, ignore_time=True)

    counts = []
    for date, df_ in df.groupby(by='date', sort=True):
        counts.append((date, df_.shape[0]))

    for date, count in counts:
        print('{}: {}'.format(date.strftime('%Y/%m/%d'), count))


if __name__ == '__main__':
    main()
