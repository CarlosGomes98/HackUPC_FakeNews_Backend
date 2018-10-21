from model.classifier import Classifier
from news_searcher.news_searcher import searcher_score


def rate(title, body):
    # Get score from trained classifier
    clf = Classifier()
    classifier_score = clf.predict(title, body)
    # Get score from article searcher
    related_article, search_score = searcher_score(title, body)
    #search_score = min(0.9, search_score * 2)

    return classifier_score * 0.7 + search_score * 0.3, related_article