"""
Refer to the following links:
- soynlp package: https://github.com/lovit/soynlp
"""
import os

import gensim
import numpy as np
import re
import time
import itertools
from collections import Counter
from konlpy import tag
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import MaxScoreTokenizer

from newstrends import utils
from newstrends.data import mysql


def preprocess(articles):
    stopwords = ['&#039;', '&quot;', '<span>', '</span>', '<span id="divTitle">',
                 '</font>', '<b>', '</b>', '포토', '등', '첫', '것', '중',
                 '국내', '정부', '전', '오늘', '종합', ]
    replace_dict = {'···': '…', '...': '…', '..': '…'}
    new_articles = []
    for article in articles:
        for word in stopwords:
            article = article.replace(word, '')
        for word, replace in replace_dict.items():
            article = article.replace(word, replace)
        article = re.sub(' +', ' ', article)
        if article.startswith(' '):
            article = article[1:]
        new_articles.append(article)
    return new_articles


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
    articles = preprocess(articles)
    model = get_tag_model(package)
    words_list = []
    for title in articles:
        words_list.append(model.nouns(title))
    return words_list


def extract_nouns_by_soynlp(articles):
    articles = preprocess(articles)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(articles)
    for noun in nouns.keys():
        print(noun)


def parse_by_soynlp(articles):
    articles = preprocess(articles)
    tokenizer = MaxScoreTokenizer()
    words_list = []
    for article in articles:
        words = tokenizer.tokenize(article)
        words_list.append(words)
    return words_list


def parser_by_minyong(articles):
    stopwords = ['없는', '것인가', '&#039;', '&quot;', '<span>', '</span>',
                 '<span id="divTitle">', '</font>', '<b>', '</b>', '포토',
                 '[포토]', '속보', '[속보]', '첫', '등', '중', '수', '외',
                 '전', '내', '것', '만에', '더', '논란', '·', '발표', '안',
                 '후', '출시', '위한', 'ET투자뉴스', '\u200b']
    li = []
    for title in articles:
        title = title.replace(',', ' ')
        title = title.replace("','", ' ')
        title = title.replace("'", '')
        title = title.replace('"', '')
        title = title.replace("'", '')
        title = title.replace('‘', '')
        title = title.replace('’', '')
        title = title.replace('[', ' ')
        title = title.replace(']', ' ')
        title = title.replace('(', ' ')
        title = title.replace(')', ' ')
        li.append([word for word in title.split() if word not in stopwords])
    return li


def main():
    entries = mysql.select_articles(field=['title', 'description', 'date'])
    dates = [e[2].date() for e in entries]
    dates = np.unique(dates)
    words_dict = dict()
    for e in entries:
        date = e[2].date()
        try:
            words_dict[date]
            words_dict[date] += [e[0]]
        except:
            words_dict[date] = [e[0]]

    parse_dict = dict()
    rank_dict = dict()
    start_time = time.time()
    f = open("./keywords.txt", 'w')

    for date in dates:
        if len(words_dict[date]) < 300:
            continue
        # parse_dict[date] = parse_by_konlpy(words_dict[date])
        parse_dict[date] = parser_by_minyong(words_dict[date])
        parse_dict[date] = list(itertools.chain(*parse_dict[date]))
        rank_dict[date] = Counter(parse_dict[date]).most_common(20)
        print(rank_dict[date], date)
        data = "{} {}\n".format(rank_dict[date], date)
        f.write(data)
    f.close()
    print(time.time() - start_time)

if __name__ == '__main__':
    main()
