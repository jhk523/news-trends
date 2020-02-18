from konlpy import tag
from soynlp.noun import LRNounExtractor_v2

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
    bad_words = ['\'', '‘', '’', '"', '“', '”', '&quot', '…', '&#039', ';', ',', '·', '...']
    for article in articles:
        new_article = article
        for w in bad_words:
            new_article = new_article.replace(w, ' ')
        new_articles.append(new_article)
    return new_articles


def parse_by_soynlp(articles):
    articles = preprocess(articles)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(articles)
    for noun in nouns.keys():
        print(noun)


def main():
    entries = mysql.select_news_articles(field=['title', 'description'])
    titles = [e[0] for e in entries]
    contents = [e[1] for e in entries]
    parse_by_soynlp(titles + contents)


if __name__ == '__main__':
    main()
