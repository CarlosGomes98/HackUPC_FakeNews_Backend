from sklearn.feature_extraction.text import CountVectorizer
import pickle
from os import path
from enum import Enum
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

tfidf_pickle_path = 'model/pickles'

stop_words = [
       "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
       "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
       "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
       "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
       "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
       "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
       "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
       "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
       "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
       "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
       "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
       "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
       "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
       "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
       "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
       "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
       "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
       "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
       "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
       "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
       "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
       "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
       "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
       "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
       "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
       ]

class Text(Enum):
    TITLE = 1
    BODY = 2
    FULL = 3


def get_full_pickle_path(text_type):
    return path.join(tfidf_pickle_path, text_type.name + '_vectorizer.sav')


def fit_tf_idf(texts_dataset, text_type):
    if path.exists(get_full_pickle_path(text_type)):
        return pickle.load(open(get_full_pickle_path(text_type), 'rb'))

    vectorizer = CountVectorizer(stop_words=stop_words)
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

def extract_tf_idf_and_word_id_map(texts_dataset, text_type=Text.BODY):
    # for testing, use vectorizer created during training
    if path.exists(get_full_pickle_path(text_type)):
        vectorizer = pickle.load(open(get_full_pickle_path(text_type), 'rb'))
    # for training, create new vectorizer
    else:
        vectorizer = fit_tf_idf(texts_dataset, text_type)
    word_id_map = vectorizer.get_feature_names()
    X = vectorizer.transform(texts_dataset)

    return X, word_id_map


def cos_similarity(titles, bodies):
    full_texts = [titles[i] + ' ' + bodies[i] for i in range(len(titles))]
    vectorizer = fit_tf_idf(full_texts, text_type=Text.FULL)
    titles_tfidf = extract_tf_idf(titles, text_type=Text.FULL)
    bodies_tfidf = extract_tf_idf(bodies, text_type=Text.FULL)
    similarities = [cosine_similarity(titles_tfidf[i, :], bodies_tfidf[i, :]) for i in range(titles_tfidf.shape[0])]
    similarities = np.asarray(similarities)
    return np.reshape(similarities, (-1, 1))

def coss_similarity(text1, text2):
    vectorizer = pickle.load(open(get_full_pickle_path(Text.FULL), 'rb'))
    text1_tfidf = extract_tf_idf(text1, text_type=Text.FULL)
    text2_tfidf = extract_tf_idf(text2, text_type=Text.FULL)


    similarity = cosine_similarity(text1_tfidf[0, :], text2_tfidf[0, :])

    return similarity[0]


def extract_features(titles, bodies):
    bodies_tfidf = extract_tf_idf(bodies)
    titles_tfidf = extract_tf_idf(titles, text_type=Text.TITLE)

    combined_tf_idf = sparse.hstack([titles_tfidf, bodies_tfidf])
    cs = sparse.coo_matrix(cos_similarity(titles, bodies))

    X = sparse.hstack([combined_tf_idf, cs])
    truncatedSVD = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
    if path.exists("model/pickles/SVD.sav"):
        svd = pickle.load(open("model/pickles/SVD.sav", 'rb'))
    else:
        svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
        svd.fit(X)
        pickle.dump(truncatedSVD, open("model/pickles/SVD.sav", "wb"))

    X = svd.transform(X)
    return X
