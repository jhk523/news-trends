from konlpy import tag

from newstrends.data import mysql


def get_tag_model(name):
    if name == 'kkma':
        model = tag.Kkma()
    elif name == 'hannanum':
        model = tag.Hannanum()
    elif name == 'komoran':
        model = tag.Komoran()
    elif name == 'mecab':
        model = tag.Mecab()
    elif name == 'okt':
        model = tag.Okt()
    else:
        raise ValueError()
    return model


def main():
    articles = mysql.select_news_articles(field='title')
    model = get_tag_model(name='komoran')
    words_list = []
    for r in articles:
        title = r[0]
        words = model.nouns(title)
        words_list.append(words)
        print(words)


if __name__ == '__main__':
    main()
