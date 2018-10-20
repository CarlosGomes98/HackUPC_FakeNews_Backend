from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from os import path
from enum import Enum
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tfidf_pickle_path = 'model/pickles'

class Text(Enum):
    TITLE = 1
    BODY = 2
    FULL = 3


def get_full_pickle_path(text_type):
    return path.join(tfidf_pickle_path, text_type.name + '_vectorizer.sav')


def fit_tf_idf(texts_dataset, text_type):
    if path.exists(get_full_pickle_path(text_type)):
        return pickle.load(open(get_full_pickle_path(text_type), 'rb'))

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


def cos_similarity(titles, bodies):
    full_texts = [titles[i] + ' ' + bodies[i] for i in range(len(titles))]
    vectorizer = fit_tf_idf(full_texts, text_type=Text.FULL)
    titles_tfidf = extract_tf_idf(titles, text_type=Text.FULL)
    bodies_tfidf = extract_tf_idf(bodies, text_type=Text.FULL)

    print("IN COSINE SIMILARITY")
    print(titles_tfidf.shape, bodies_tfidf.shape)

    similarities = [cosine_similarity(titles_tfidf[i, :], bodies_tfidf[i, :]) for i in range(titles_tfidf.shape[0])]
    print("Similarities", len(similarities))
    similarities = np.asarray(similarities)
    return np.reshape(similarities, (-1, 1))


def extract_features(titles, bodies):
    bodies_tfidf = extract_tf_idf(bodies)
    titles_tfidf = extract_tf_idf(titles, text_type=Text.TITLE)

    combined_tf_idf = sparse.hstack([titles_tfidf, bodies_tfidf])
    cs = sparse.coo_matrix(cos_similarity(titles, bodies))
    print("Cosine shape", cs.shape)

    X = sparse.hstack([combined_tf_idf, cs])
    return X
