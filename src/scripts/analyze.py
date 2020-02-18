"""
Refer to the following links:
- soynlp package: https://github.com/lovit/soynlp
"""
import os

import gensim
import numpy as np
from konlpy import tag
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import MaxScoreTokenizer

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


def parse_by_konlpy(articles):
    models = [get_tag_model(name) for name in ('kkma', 'hannanum', 'komoran', 'okt')]
    for title in articles:
        print(title)
        for i, model in enumerate(models):
            words = model.nouns(title)
            print(f'{i + 1}. {words}')
        print()


def preprocess(articles):
    new_articles = []
    bad_words = ['\'', '‘', '’', '"', '“', '”', '&quot', '…', '&#039', ';', ',', '·', '...', '[', ']', '\\u200b', '?']
    for article in articles:
        new_article = article
        for w in bad_words:
            new_article = new_article.replace(w, ' ')
        new_articles.append(new_article)
    return new_articles


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


def main():
    entries = mysql.select_news_articles(field=['title', 'description'])
    titles = [e[0] for e in entries]
    contents = [e[1] for e in entries]

    words_list = parse_by_soynlp(titles)
    words_all = list(set([w for words in words_list for w in words]))
    word_dict = gensim.models.word2vec.Word2Vec(words_list, size=2)

    words, vectors = [], []
    for w in words_all:
        if w in word_dict:
            words.append(w)
            vectors.append(word_dict[w])
    words = np.array(words, dtype=str)
    vectors = np.array(vectors, dtype=np.float32)

    out_path = '../out'
    os.makedirs(out_path, exist_ok=True)
    np.savetxt(os.path.join(out_path, 'words.tsv'), words, fmt='%s', delimiter='\t')
    np.savetxt(os.path.join(out_path, 'vectors.tsv'), vectors, delimiter='\t')


if __name__ == '__main__':
    main()
