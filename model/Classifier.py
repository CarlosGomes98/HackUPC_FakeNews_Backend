from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import feature_extractor
import fetch_dataset
import pickle
# pandas dataframe containing title, text and label(0 for true 1 for fake)
dataset = fetch_dataset.read_dataset("data/train.csv")
train_set, test_set = train_test_split(dataset, test_size=0.2)

train_titles = train_set[:, 0]
train_bodies = train_set[:, 1]
Y_train = train_set[:, 2]
Y_train = Y_train.astype('int')
test_titles = test_set[:, 0]
test_bodies = test_set[:, 1]
Y_test = test_set[:, 2]

X_train = feature_extractor.extract_features(train_titles, train_bodies.tolist())

print(Y_train.shape)
print(X_train.shape)
reg = linear_model.LogisticRegression(verbose=1)
reg.fit(X_train, Y_train)

print(accuracy_score(reg.predict(feature_extractor.extract_features(test_titles, test_bodies.tolist())), Y_test.astype("int")))

pickle.dump(reg, open("pickles/regression.sav", 'wb'))

