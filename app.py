from flask import Flask, jsonify
from flask import request
from flask_restful import Resource, Api

#code which helps initialize our server

app =  Flask(__name__)
api = Api(app)

# from stock_database.parse_csv import Migration
import os

#migration = Migration()
port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)  