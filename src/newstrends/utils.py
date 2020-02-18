def preprocess(articles):
    new_articles = []
    stopwords = ['\'', '‘', '’', '"', '“', '”', '&quot', '…', '&#039', ';', ',',
                 '·', '...', '[', ']', '\\u200b', '?', '(', ')']
    for article in articles:
        new_article = article
        for w in stopwords:
            new_article = new_article.replace(w, ' ')
        new_articles.append(new_article)
    return new_articles
