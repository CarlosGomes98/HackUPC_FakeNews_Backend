import pandas as pd


def read_dataset(path):
    dataset = pd.read_csv(path)
    dataset = dataset.drop("id", axis=1)
    dataset = dataset.drop("author", axis=1)
    dataset = dataset.dropna(axis=0)
    dataset["text"] = dataset["text"].str.replace("\n", "")
    dataset = dataset.values
    print(dataset.shape)
    return dataset
