from flask import Flask
from flask_restful import Resource, Api, reqparse
from newspaper import Article

class ArticleParser:
    def __init__ (self,url):
      self.url = url
      self.article = Article(self.url)
      self.article.download()

    def parse(self):
        self.article.parse()
        data = {}
        
        # cleaning parsed data and storing it in a dict
        rep=["[","u'","\\n","\n"]
        for r in rep:
            data['body'] = self.article.text.replace(r,'')
            data['title'] = self.article.title.replace(r,'')
            data['desc'] = self.article.meta_description.replace(r,'')
            data['src'] = self.article.source_url.replace(r,'')
        data['authors'] = self.article.authors
        data['all'] = self.article

        return data




class fake_o_meter(Resource):
    def post(self):

        post_parser = reqparse.RequestParser()
        post_parser.add_argument('url')
        args = post_parser.parse_args()

        url = args['url']

        article_parser = ArticleParser(url)
        parsed_data =  article_parser.parse()

        return {
            'STATUS': 'WIP - No models yet',
            'title':parsed_data['title'], 
            'body':parsed_data['body'],
            'desc': parsed_data['desc'],
            'src': parsed_data['src'],
            'authors': parsed_data['authors'],
            'all': parsed_data['all'].meta_data
        }


app = Flask(__name__)
api = Api(app)

api.add_resource(fake_o_meter, '/')

if __name__ == '__main__':
    app.run(debug=True)