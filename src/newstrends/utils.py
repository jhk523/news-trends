import re


def preprocess(articles):
    new_articles = []
    for article in articles:
        new_article = article
        new_article = re.sub(' +', ' ', new_article)
        if new_article.startswith(' '):
            new_article = new_article[1:]
        new_articles.append(new_article)
    return new_articles
