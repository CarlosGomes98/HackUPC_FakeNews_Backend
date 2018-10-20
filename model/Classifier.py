from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import path
import feature_extractor
import fetch_dataset
import pickle
import numpy as np

tfidf_pickle_path = path.join('model', 'pickles')


def get_model_pickle_path():
    return path.join(tfidf_pickle_path, "regression.sav")


class Classifier:

    def __init__(self):
        if path.exists(get_model_pickle_path()):
            self.clf = pickle.load(open(get_model_pickle_path(), 'rb'))

    def train(self):

        dataset = fetch_dataset.read_dataset("model/data/train.csv")
        train_set, test_set = train_test_split(dataset, test_size=0.2)

        train_titles = train_set[:, 0].tolist()
        train_bodies = train_set[:, 1].tolist()
        Y_train = train_set[:, 2]
        Y_train = Y_train.astype('int')
        test_titles = test_set[:, 0].tolist()
        test_bodies = test_set[:, 1].tolist()
        Y_test = test_set[:, 2].astype('int')

        X_train = feature_extractor.extract_features(train_titles, train_bodies)

        print(X_train.shape, Y_train.shape)
        reg = linear_model.LogisticRegression(verbose=1)
        reg.fit(X_train, Y_train)

        X_test = feature_extractor.extract_features(test_titles, test_bodies)

        print("Accuracy", accuracy_score(reg.predict(X_test), Y_test))

        pickle.dump(reg, open(get_model_pickle_path(), 'wb'))
        self.clf = reg

    def predict(self, title, body):
        features = feature_extractor.extract_features([title], [body])

        return self.clf.predict_proba(features)


