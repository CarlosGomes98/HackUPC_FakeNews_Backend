from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from os import path
from enum import Enum


tfidf_pickle_path = 'pickles'

class Text(Enum):
    TITLE = 1
    BODY = 2


def get_full_pickle_path(text_type):
    return path.join(tfidf_pickle_path, text_type.name + '_vectorizer.sav')


def fit_tf_idf(texts_dataset, text_type):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts_dataset)
    pickle.dump(vectorizer, open(get_full_pickle_path(text_type), 'wb'))
    
    return vectorizer


def extract_tf_idf(texts_dataset, text_type=Text.BODY):
    # for testing, use vectorizer created during training
    if path.exists(get_full_pickle_path(text_type)):
        vectorizer = pickle.load(open(get_full_pickle_path(text_type), 'rb'))
    # for training, create new vectorizer
    else:
        vectorizer = fit_tf_idf(texts_dataset, text_type)

    X = vectorizer.transform(texts_dataset)

    return X


def extract_features(titles, bodies):
    bodies_tfidf = extract_tf_idf(bodies)
    titles_tfidf = extract_tf_idf(titles, text_type=Text.TITLE)

    return bodies_tfidf
