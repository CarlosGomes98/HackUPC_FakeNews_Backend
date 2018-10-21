from model.classifier import Classifier
from news_searcher.news_searcher import searcher_score
from model.whitelist import is_from_trusted_source


def rate(title, body, source):
    # Get score from trained classifier
    clf = Classifier()
    classifier_score = clf.predict(title, body)
    # Get score from article searcher
    related_article, search_score = searcher_score(title, body)
    #search_score = min(0.9, search_score * 2)
    if source[:7] == "http://":
        source = source[7:]
    elif source[:8] == "https://":
        source = source[8:]

    first_slash_index = source.find("/")
    if first_slash_index != -1:
        source = source[:first_slash_index]

    print(source)
    is_trusted = is_from_trusted_source(source)
    print("Is trusted ", is_trusted)
    p = 0.55
    q = 0.3

    if is_trusted:
        return 10*(classifier_score * p + (1 - p - q) + search_score * q), related_article
    else:
        return 10*(classifier_score * 0.7 + search_score * 0.3), related_article