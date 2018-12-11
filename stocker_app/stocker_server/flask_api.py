from flask import Flask, jsonify, request
from flask_restful import Resource
import pickle, json
from stocker_app.config import configs
from stocker_app.stocker_logic.stock_model import SModel
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stocker_server.tasks import predict, migrate_data
from stocker_app.stock_database.migration.parse_csv import Migration
from stocker_app.models import Prediction, PredictionParam
from stocker_app.stock_database.DAO import DAO

model_path = configs['model_path']
dao = DAO()

class WelcomePage(Resource):
    def get(self):
        return 'Welcome to Stock Prediction Tool, create by Nguyen Quoc Trung & Nguyen Thi Hien'

class PriceData(Resource):
    def get(self, ticker):
        start_date = request.args.get('start-date')
        stock = FinancialData(ticker=ticker)
        result = stock.get_data(start_date).tail(100)
        if result.empty == True:
            return []
        result = {
            'ticker': stock.ticker,
            'max_date': stock.max_date,
            'min_date': stock.min_date,
            'max_price': stock.max_price,
            'min_price':stock.min_price,
            'price': result[["ds", "Volume","Open","High","Low","Close"]].to_dict('records')
        }
        return jsonify(result)

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

class PredictionAPI(Resource):
    def get(self):
        model_id = request.args.get('model-id', None)
        if model_id == None:
            return 'Invalid Model ID'

        predictions = {}
        status = {}
        params_db = dao.get_prediction_model(model_id = model_id)
        params = PredictionParam()
        params.set_params(params=params_db)
        try:
            status = dao.get_model_status(model_id=model_id)
            message = ''
            if status['status'] == 0:
                message = 'Please build the model first /model/build'
            elif status['status'] == 2:
                message = 'Model is being built, please try again'
            elif status['status'] == 1:
                predictions = predict.delay(params=params.get_dict())
                result = predictions.wait()
                return result

            return {
                'model_id': model_id,
                'status_code':status,
                'message': message 
            }
        except Exception as ex:
            return {
                'message': message,
                'exception': ex.__dict__,
                'modelid': model_id,
                'params': params.get_dict(),
                'status': status
            }

class ModelBuild(Resource):
    def post(self, ticker):
        start_date = request.args.get('start-date', '2016-08-01')
        lag = request.args.get('lag', 1)
        prior = request.args.get('prior', 0.05)
        seasonalities = request.args.get('seasonalities', 'm-q-y')
        status = {}

        try:
            lag =int(lag)
            prior = float(prior)
            params = PredictionParam(seasonalities = seasonalities, changepoint_prior_scale=prior, ticker=ticker, lag = lag, date=start_date)
            model_id = params.get_hash()
           
            status = dao.get_model_status(model_id=model_id)
            resp_status = 'None'
            if status['status'] == 0:
                dao.update_model_status(model_id = model_id, status=2)
                predict.delay(params=params.get_dict())
                resp_status = 'started'
            elif status['status'] == 1:
                resp_status= 'finished'
            elif status['status'] == 2:
                resp_status = 'in progress'
            return {
                'status': resp_status,
                'status_code': status
            }
        except Exception as ex:
            return {
                'message': 'Bad Request',
                'exception': ex
            }


class ModelStatus(Resource):
    def get(self, model_id):
        status = dao.get_model_status(model_id=model_id)
        return status

class ModelList(Resource):
        
    def get(self):
        models = dao.get_prediction_models().drop('model_pkl', axis=1)
        models =  models.to_dict('records')
        return jsonify(models)

class DataMigration(Resource):
    def get(self, start):
        try:
            migration = Migration()
            inprogress = migration.get_migration_status() == 1
            current_index= migration.get_current_migration_progress()
        except Exception as ex:
            return ex
        finally:
            return {
                'in_progress': inprogress,
                'latest_index': current_index
            }

    def post(self, start):
        try:
            migration = Migration()
            status = migration.get_migration_status()
            if status != start:
                migration.set_setting(key='migration', value=start)
            if status == 0 and start == 1:
                process = migrate_data.delay(start=start)
        except Exception as ex:
            print(ex)
            return 'Error'
        signal = 'STOP' if start == 0 else 'START' 
        return 'Signal %s sent' %signal