from flask import Flask, jsonify
from flask import request
from flask_restful import Resource, Api

#code which helps initialize our server

app =  Flask(__name__)
api = Api(app)