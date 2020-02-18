def preprocess(articles):
    new_articles = []
    bad_words = ['\'', '‘', '’', '"', '“', '”', '&quot', '…', '&#039', ';', ',', '·', '...', '[', ']', '\\u200b', '?']
    for article in articles:
        new_article = article
        for w in bad_words:
            new_article = new_article.replace(w, ' ')
        new_articles.append(new_article)
    return new_articles
