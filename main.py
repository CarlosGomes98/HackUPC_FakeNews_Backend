from model.classifier import Classifier
from news_searcher.news_searcher import searcher_score


def rate(title, body):
    # Get score from trained classifier
    clf = Classifier()
    classifier_score = clf.predict(title, body)

    # Get score from article searcher
    search_score = searcher_score(title, body)

    return classifier_score, search_score


print(rate("Spain makes too much paella for the people to handle", "Catalonia declares independence from Spain."))