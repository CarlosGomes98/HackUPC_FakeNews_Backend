from model.feature_extractor import extract_tf_idf_and_word_id_map, cos_similarity
import requests
import numpy as np
import pandas as pd
from scipy import sparse
from article_parser.article_parser import ArticleParser

api_key = 'e919320c855c40638b2534863e3974df'


def search_articles(query):
    url = ('https://newsapi.org/v2/everything?'
           'q={}&'.format(query) +
           'sortBy=popularity&'
           'apiKey=e919320c855c40638b2534863e3974df')

    response = requests.get(url)
    results = []
    for entry in response.json()["articles"][:5]:
        results.append({'title': entry['title'], 'url': entry['url'], 'image_url': entry['urlToImage']})
    return results


def searcher_score(title, body):

    tf_idf, word_id_map = extract_tf_idf_and_word_id_map([body])
    tf_idf = np.transpose(tf_idf.todense())
    sorted_indices = tf_idf.argsort(axis=0)[::-1]
    top_3_words = []
    for i in range(3):
        top_3_words.append(word_id_map[sorted_indices[i, 0]])
    print(top_3_words)

    print("Calculating search score -----------------------------")

    print("Searching for articles from google........")
    print(" ".join(top_3_words))
    results = search_articles(" ".join(top_3_words))

    print("Fetching closest articles.......")
    similarities = []
    for article in results:
        result = ArticleParser(article["url"]).parse()
        similarities.append(cos_similarity([title + " " + result["title"]], [body + " " + result["body"]]))

    max = 0
    for i in range(len(similarities)):
        if similarities[i] > max:
            max = i

    print("Finding similarity with found article")

    print("Highest similarity is ", results[max]["title"], " with ", similarities[max])
    print(similarities[max][0][0])
    return results[max], similarities[max][0][0]