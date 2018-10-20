from flask import Flask
from flask_restful import Resource, Api, reqparse
from parser import ArticleParser
from model.classifier import Classifier
from  news_searcher.news_searcher import searcher_score
from main import rate

class fake_o_meter(Resource):
    def post(self):

        post_parser = reqparse.RequestParser()
        post_parser.add_argument('url')
        args = post_parser.parse_args()

        url = args['url']

        article_parser = ArticleParser(url)
        parsed_data = article_parser.parse()

        score = rate(parsed_data["title"], parsed_data["body"])


        return score


app = Flask(__name__)
api = Api(app)

api.add_resource(fake_o_meter, '/')

if __name__ == '__main__':
    app.run(debug=True)