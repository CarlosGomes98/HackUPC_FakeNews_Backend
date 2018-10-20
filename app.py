from flask import Flask
from flask_restful import Resource, Api, reqparse
from article_parser.article_parser import ArticleParser
from main import rate
import os

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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='127.0.0.1', port=port)