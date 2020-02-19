import re


def preprocess(articles):
    stopwords = ['&#039;']
    new_articles = []
    for article in articles:
        for word in stopwords:
            article = article.replace(word, '')
        article = re.sub(' +', ' ', article)
        if article.startswith(' '):
            article = article[1:]
        new_articles.append(article)
    return new_articles
