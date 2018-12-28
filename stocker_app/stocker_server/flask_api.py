from flask import Flask, jsonify, request
from flask_restful import Resource
import pickle, json
from stocker_app.config import configs
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stocker_server.tasks import predict, migrate_data, build_model, evaluate_prediction
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
        result = stock.get_data(start_date= start_date)
        if result.empty == True:
            return []
        result = {
            'info':{
                'ticker': stock.ticker,
                'max_date': stock.max_date,
                'min_date': stock.min_date,
                'max_price': stock.max_price,
                'min_price':stock.min_price,
            },
            'price': result[["ds", "volume","open","high","low","close"]].to_dict('records')
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
        results = stock.get_moving_average(lags=lags, start_date=start_date)
        json=dict()
        for (key, result) in results.items():
            json[key]=result.to_dict('records')
        return  jsonify(json)

class PredictionAPI(Resource):
    def get(self):
        model_id = request.args.get('model-id', None)
        days = request.args.get('days', 30)
        if model_id == None:
            return 'Invalid Model ID'

        print('MODEL FUCKING ID', model_id)

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
                predictions = predict.delay(params=params.get_dict(),days= days )
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

class PredictionEvaluation(Resource):
    def get(self):
        model_id = request.args.get('model-id', None)
        evaluation_end = request.args.get('evaluation-end', None)
        if model_id == None:
            return 'Invalid Model ID'

        print('MODEL FUCKING ID', model_id)

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
                evaluation = evaluate_prediction.delay(params=params.get_dict(),evaluation_end= evaluation_end )
                result = evaluation.wait()
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
        status = {}
        try:
            start_date = request.args.get('start-date', None)
            label = request.args.get('label', 'close')
            lag = request.args.get('lag', 1)
            prior = request.args.get('prior', 0.05)
            seasonalities = request.args.get('seasonalities', 'm-q-y')
            training_years = request.args.get('training-years', 1.0)
            status = {}

            lag =int(lag)
            prior = float(prior)
            training_years = float(training_years)
            params = PredictionParam(
                seasonalities = seasonalities, 
                changepoint_prior_scale=prior, 
                ticker=ticker, 
                lag = lag, 
                date=start_date,
                label=label,
                training_years=training_years
                )
            
            model_id = params.get_hash()
            print('MODEL PPARAMS', params.get_description())
           
            status = dao.get_model_status(model_id=model_id)
            resp_status = 'None'
            if status['status'] == 0:
                dao.update_model_status(model_id = model_id, status=2)
                build_model.delay(params=params.get_dict())
                resp_status = 'started'
            elif status['status'] == 1:
                resp_status= 'finished'
            elif status['status'] == 2:
                resp_status = 'in progress'
            elif status['status'] == -2:
                resp_status = 'Parameter invalid, model was not built'
            return {
                'status': resp_status,
                'status_code': status
            }
        except Exception as ex:
            return {
                'message': 'Bad Request',
                'exception': str(ex),
                'status': status
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