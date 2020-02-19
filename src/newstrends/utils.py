import re


def preprocess(articles):
    stopwords = ['&#039;', '&quot;', '<span>', '</span>', '<span id="divTitle">',
                 '</font>', '<b>', '</b>']
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
