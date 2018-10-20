from model.classifier import Classifier
from news_searcher.news_searcher import searcher_score


def rate(title, body):
    # Get score from trained classifier
    clf = Classifier()
    classifier_score = clf.predict(title, body)

    # Get score from article searcher
    search_score = searcher_score(title, body)

    return classifier_score * 0.8 + search_score * 0.2