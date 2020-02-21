from collections import Counter, defaultdict
from datetime import datetime

from newstrends import utils
from newstrends.data import mysql


def main():
    entries = mysql.select_articles(
        field=['title', 'description', 'date'],
        date_from=datetime(year=2020, month=2, day=11))
    words_dict = defaultdict(lambda: [])
    for e in entries:
        words_dict[e[2].date()].append(e[0])

    rank_dict = {}
    for date in sorted(words_dict.keys()):
        keywords = utils.to_keywords(words_dict[date])
        keywords = [w for words in keywords for w in words]
        rank_dict[date] = Counter(keywords).most_common(20)
        print('{}: {} {}'.format(date, rank_dict[date], date))


if __name__ == '__main__':
    main()
