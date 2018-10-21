from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import path
from model.feature_extractor import extract_features
from model.fetch_dataset import read_dataset
import pickle

tfidf_pickle_path = path.join('model', 'pickles')


def get_model_pickle_path():
    return path.join(tfidf_pickle_path, "regression.sav")


class Classifier:

    def __init__(self):
        self.clf = None
        if path.exists(get_model_pickle_path()):
            self.clf = pickle.load(open(get_model_pickle_path(), 'rb'))

    def has_model(self):
        return self.clf is not None

    def train(self):

        dataset = read_dataset("data/train.csv")
        train_set, test_set = train_test_split(dataset, test_size=0.2)

        train_titles = train_set[:, 0].tolist()
        train_bodies = train_set[:, 1].tolist()
        Y_train = train_set[:, 2]
        Y_train = Y_train.astype('int')
        test_titles = test_set[:, 0].tolist()
        test_bodies = test_set[:, 1].tolist()
        Y_test = test_set[:, 2].astype('int')

        X_train = extract_features(train_titles, train_bodies)
        reg = linear_model.LogisticRegression(verbose=1)
        reg.fit(X_train, Y_train)

        X_test = extract_features(test_titles, test_bodies)

        print("Accuracy", accuracy_score(reg.predict(X_test), Y_test))

        pickle.dump(reg, open(get_model_pickle_path(), 'wb'))
        self.clf = reg

    def predict(self, title, body):
        if not self.has_model():
            self.train()

        features = extract_features([title], [body])
        trust_score = self.clf.predict_proba(features)[0][0]
        return round(trust_score, 2)