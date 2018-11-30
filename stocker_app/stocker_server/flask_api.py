from flask import Flask, jsonify, request
from flask_restful import Resource
import pickle
import json
from stocker_app.config.setting import configs
from stocker_app.stocker_logic.stock_model import SModel
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stocker_server import tasks as tasks

model_path = configs['model_path']

model = pickle.load(open("%s/prophet_model_MA_1_Close.pkl" %model_path, "rb"))

class Hello(Resource):
    def get(self):
        return 'hello'

class PriceData(Resource):
    def get(self, ticker):
        start_date = request.args.get('start-date')
        stock = FinancialData(ticker=ticker)
        result = stock.get_data(start_date)
        if result.empty == True:
            return []
        result = result[["ds", "Volume","Open","High","Low","Close"]]
        return jsonify(result.to_dict('records'))

class MovingAverage(Resource):
    def get(self, ticker):
        start_date = request.args.get('start-date')
        lags = request.args.get('lags')
        try:
            lags =map(int, lags.split('-'))
        except Exception as ex:
            print ('No lag input', ex)
            lags= [5]
        
        print(start_date, lags)

        stock = FinancialData(ticker=ticker)
        results = stock.get_moving_averages(lags=lags, start_date=start_date)
        json=dict()
        for (key, result) in results.items():
            json[key]=result.to_dict('records')
        return  jsonify(json)

class Prediction(Resource):
    def get(self, ticker):
        start_date = request.args.get('start-date')
        lags = request.args.get('lags')
        try:
            lags =map(int, lags.split('-'))
        except Exception as ex:
            print ('No lag input', ex)
            lags= [1]
        finally:
            # Future dataframe with specified number of days to predict
            import stocker_app.stocker_server.tasks as tasks
            predictions = tasks.predict.delay(ticker=ticker, lags=lags, start_date=start_date)
            result = predictions.wait()
            print(result)
        return result

class DataMigration(Resource):
    def get(self, start):
        try:
            process = tasks.migrate_data.delay(start=start)
        except Exception as ex:
            return ex     
        return 'DataMigration started'

# @app.route('/describe', methods=["GET"])
# def describe():
#     result = stock.describe_stock()
#     return jsonify(result)

# @app.route('/price/<string:ticker>', methods = ["GET"])
# def get_stock_data(ticker):
#     return stock.get_data().to_json(orient="records")
# #defining a /hello route for only post requests
# @app.route('/future', methods=['GET'])
# def index():
    

# # if __name__ == '__main__':
# #     app.run(debug=True)