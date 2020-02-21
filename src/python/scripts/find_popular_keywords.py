from newstrends import utils


def main():
    df = utils.find_popular_keywords()
    for date, df_ in df.groupby(by='date'):
        print(f'{date}:', end=' ')
        for _, row in df_.iterrows():
            print('{} ({})'.format(row['word'], row['count']), end=' ')
        print()


if __name__ == '__main__':
    main()
