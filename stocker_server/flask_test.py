from flask import jsonify
from flask import request
from flask_restful import Resource
from stocker_server import api
import pickle
from global_configs import configs
from stocker_logic.stock_model import SModel
from stock_database.financial_data import FinancialData
from stocker_server.endpoints import endpoints
import json

data_endpoint = endpoints['price_data']
ma_endpoint = endpoints['moving_average_data']


model_path = configs['model_path']


model = pickle.load(open("%s/prophet_model_MA_1_Close.pkl" %model_path, "rb"))
class PriceData(Resource):
    def get(self, ticker):
        start_date = request.args.get('start-date')
        stock = FinancialData(ticker=ticker)
        result = stock.get_data(start_date)
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

api.add_resource(PriceData, '%s/<string:ticker>' %data_endpoint)
api.add_resource(MovingAverage, '%s/<string:ticker>' %ma_endpoint)

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
#      # Future dataframe with specified number of days to predict
#     predicted = model.make_future_dataframe(periods=30, freq='D')
#     predicted = model.predict(predicted)
#     # Only concerned with future dates
#     future = predicted[predicted['ds'] >= stock.max_date.date()]

#     # Remove the weekends
#     #future = self.remove_weekends(future)

#     future = future.dropna()
    
#     # Calculate whether increase or not
#     predicted['diff'] = predicted['yhat'].diff()
#     future['diff'] = future['yhat'].diff()

#     predicted = predicted.dropna()

#     # Find the prediction direction and create separate dataframes
#     future['direction'] = (future['diff'] > 0) * 1
#     predicted['direction'] = (predicted['diff']> 0) *1
#     predicted['y'] = predicted['yhat']
#     result = dict()
#     result['predicted'] = predicted.to_json()
#     result['future'] = future.to_json()
    
#     return predicted[["y", "ds"]].head().to_json()

# # if __name__ == '__main__':
# #     app.run(debug=True)