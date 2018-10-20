import requests
from article_parser.article_parser import ArticleParser

api_key = 'e919320c855c40638b2534863e3974df'


def search_articles(query):
    url = ('https://newsapi.org/v2/everything?'
           'q={}&'.format(query) +
           'sortBy=popularity&'
           'apiKey=e919320c855c40638b2534863e3974df')

    response = requests.get(url)
    results = []
    for entry in response.json()["articles"][:1]:
        results.append({'title': entry['title'], 'url': entry['url'], 'image_url': entry['urlToImage']})
    return results


def searcher_score(title, body):
    print("Calculating search score -----------------------------")

    print("Searching for articles from google........")
    results = search_articles(title)
    print(results)

    print("Fetching closest articles.......")
    print(ArticleParser(results[0]["url"]).parse())

    print("Finding similarity with found article")


    return 0
