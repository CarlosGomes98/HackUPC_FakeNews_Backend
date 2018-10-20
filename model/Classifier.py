from sklearn import linear_model
from sklearn.model_selection import train_test_split
import feature_extractor
import fetch_dataset

# pandas dataframe containing title, text and label(0 for true 1 for fake)
dataset = fetch_dataset.read_dataset("data/train.csv")
print(dataset.shape)
train_set, test_set = train_test_split(dataset, test_size=0.2)

print(train_set.shape)
# print(test_set)s

train_titles = train_set[:, 0]
train_bodies = train_set[:, 1]
train_labels = train_set[:, 2]

test_titles = test_set[:, 0]
test_bodies = test_set[:, 1]
test_labels = test_set[:, 2]

features = feature_extractor.extract_features(train_titles, train_bodies.tolist())
print(features)
#reg = linear_model.LinearRegression()

