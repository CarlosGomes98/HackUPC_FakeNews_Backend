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
    print(results)

    print("Fetching closest articles.......")
    print(ArticleParser(results[1]["url"]).parse())
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
    return results[max], similarities[max]

searcher_score("", "Donald Trump has confirmed the US will leave an arms control treaty with Russia dating from the cold war that has kept nuclear missiles out of Europe for three decades. Trump says he'd prefer to choose woman as US ambassador to UN Read more “We’ll have to develop those weapons,” the president told reporters in Nevada after a rally. “We’re going to terminate the agreement and we’re going to pull out.” Trump was referring to the 1987 Intermediate-range Nuclear Forces treaty (INF), which banned ground-launch nuclear missiles with ranges from 500km to 5,500km. Signed by Ronald Reagan and Mikhail Gorbachev, it led to nearly 2,700 short- and medium-range missiles being eliminated, and an end to a dangerous standoff between US Pershing and cruise missiles and Soviet SS-20 missiles in Europe. The Guardian reported on Friday that Trump’s third national security adviser, John Bolton, a longstanding opponent of arms control treaties, was pushing for US withdrawal. The US says Russia has been violating the INF agreement with the development and deployment of a new cruise missile. Under the terms of the treaty, it would take six months for US withdrawal to take effect. US hawks have also argued that the INF treaty ties the country’s hands in its strategic rivalry with China in the Pacific, with no response to Chinese medium-range missiles that could threaten US bases, allies and shipping. Bolton and the top arms control adviser in the National Security Council (NSC), Tim Morrison, are also opposed to the extension of another major pillar of arms control, the 2010 New Start agreement with Russia, which limited the number of deployed strategic warheads on either side to 1,550. That agreement, signed by Barack Obama and Dmitri Medvedev, then president of Russia, is due to expire in 2021.")