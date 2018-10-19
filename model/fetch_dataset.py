import pandas as pd


def read_dataset():
    dataset = pd.read_csv("data/train.csv")
    dataset = dataset.drop("id", axis=1)
    print(dataset.columns)
    dataset["text"] = dataset["text"].str.replace("\n", "")

    titles = [title for title in dataset["title"]]

    bodies = [body for body in dataset["text"]]

    labels = [label for label in dataset["label"]]
    return (titles, bodies, labels)
    