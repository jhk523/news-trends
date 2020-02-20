"""
Refer to the following links:
- soynlp package: https://github.com/lovit/soynlp
"""
import os

import gensim
import numpy as np
import time
import itertools
from collections import Counter
from konlpy import tag
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import MaxScoreTokenizer

from newstrends import utils
from newstrends.data import mysql


def get_tag_model(name):
    if name == 'kkma':
        model = tag.Kkma()
    elif name == 'hannanum':
        model = tag.Hannanum()
    elif name == 'komoran':
        model = tag.Komoran()
    elif name == 'okt':
        model = tag.Okt()
    else:
        raise ValueError()
    return model


def parse_by_konlpy(articles, package='hannanum'):
    articles = utils.preprocess(articles)
    model = get_tag_model(package)
    words_list = []
    for title in articles:
        words_list.append(model.nouns(title))
    return words_list


def extract_nouns_by_soynlp(articles):
    articles = utils.preprocess(articles)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(articles)
    for noun in nouns.keys():
        print(noun)


def parse_by_soynlp(articles):
    articles = utils.preprocess(articles)
    tokenizer = MaxScoreTokenizer()
    words_list = []
    for article in articles:
        words = tokenizer.tokenize(article)
        words_list.append(words)
    return words_list


def main():
    entries = mysql.select_articles(field=['title', 'description', 'date'])
    entries = entries[:1000]
    dates = [e[2].date() for e in entries]
    dates = np.unique(dates)
    words_dict = dict()
    for e in entries:
        date = e[2].date()
        try:
            words_dict[date]
            words_dict[date] += [e[0]] + [e[1]]
        except:
            words_dict[date] = [e[0]] + [e[1]]

    parse_dict = dict()
    rank_dict = dict()
    start_time = time.time()
    for date in dates:
        parse_dict[date] = parse_by_konlpy(words_dict[date])
        # parse_dict[date] = parse_by_soynlp(words_dict[date])
        parse_dict[date] = list(itertools.chain(*parse_dict[date]))
        rank_dict[date] = Counter(parse_dict[date])
        print(rank_dict[date], date)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
