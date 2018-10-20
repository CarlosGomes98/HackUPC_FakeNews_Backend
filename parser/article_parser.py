from newspaper import Article


class ArticleParser:
    def __init__(self, url):
        self.url = url
        self.article = Article(self.url)
        self.article.download()

    def parse(self):
        self.article.parse()
        data = {}

        # cleaning parsed data and storing it in a dict
        rep = ["[", "u'", "\\n", "\n"]
        for r in rep:
            data['body'] = self.article.text.replace(r, '')
            data['title'] = self.article.title.replace(r, '')
            data['desc'] = self.article.meta_description.replace(r, '')
            data['src'] = self.article.source_url.replace(r, '')
        data['authors'] = self.article.authors
        data['all'] = self.article

        return data
