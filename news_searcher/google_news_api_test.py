import requests

api_key = 'e919320c855c40638b2534863e3974df'


def get_latest_headlines():
    url = ('https://newsapi.org/v2/top-headlines?'
           'country=us&'
           'apiKey=e919320c855c40638b2534863e3974df')
    response = requests.get(url)
    print(response.content)

    return response.json()


def search_articles(title):
    url = ('https://newsapi.org/v2/everything?'
           'q={}&'.format(title) +
           'from=2018-10-20&'
           'sortBy=popularity&'
           'apiKey=e919320c855c40638b2534863e3974df')

    response = requests.get(url)

    print(response.json()["articles"][:3])


search_articles('Donald Trump deal with South Korea')