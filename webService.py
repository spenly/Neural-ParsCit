#This shows a demo of how to use Neural ParsCit as web service for 
#enterprise systems. It's encouraged to used as service since it uses
#a lot of memory due to  word embeddings
from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.ext.jsonpify import jsonify
from time import gmtime, strftime
from Predictor import Predictor

app = Flask(__name__)
api = Api(app)
_predictor = Predictor()

class Welcome(Resource):
    def get(self):
        d = dict()
        d['status'] = 'It\'s Working!'
        result = {'result': [d]}
        return jsonify(result)

class Parscit(Resource):
    def get(self):
        d = dict()
        d['input_string'] = request.args.get('text')
        d['parsed_string'] = _predictor.parseString(d['input_string'])
        result = {'result': [d]}
        return jsonify(result)
        
api.add_resource(Parscit, '/parscit')
api.add_resource(Welcome, '/status')

if __name__ == '__main__':
     app.run(host='0.0.0.0',port='5002')
