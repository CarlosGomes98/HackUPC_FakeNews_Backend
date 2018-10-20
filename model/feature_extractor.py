from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from os import path


tfidf_pickle_path = 'pickles/vectorizer.sav'


def fill_it_up_real_nice(x):
    return x.todense()


def fit_tf_idf(texts_dataset):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    pickle.dump(vectorizer, open(tfidf_pickle_path, 'wb'))

    return vectorizer


def extract_tf_idf(texts_dataset, doIFillItUpRealNice=False):
    # for testing, use vectorizer created during training
    if path.exists(tfidf_pickle_path):
        vectorizer = pickle.load(open(tfidf_pickle_path, 'rb'))
    # for training, create new vectorizer
    else:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        pickle.dump(vectorizer, open(tfidf_pickle_path, 'wb'))

    X = vectorizer.transform(texts_dataset)
    if doIFillItUpRealNice:
        X = X.todense()

    print(X)


def extract_features(titles, bodies):
    bodies_tfidf = extract_tf_idf(bodies)

    return bodies_tfidf


corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
new_sentence = 'This is the third document'
extract_tf_idf([corpus[0]])



