from stocker_app.stocker_logic.stock_model import SModel
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stock_database.migration.parse_csv import Migration
from stocker_app.models import PredictionParam
from flask import jsonify
from stocker_app import celery
from stocker_app.stock_database.DAO import DAO
from stocker_app.utils import get_prediction_model_hash

dao = DAO()

@celery.task()
def predict(params, days):
    # only json can be passed to celery task, convert params json to params object
    __params = PredictionParam()
    result = __params.set_params_dict(params=params)
    if result == False:
        return 'Error'
    # print('[Predict Task] task received with params\n', __params.get_description())
    # stock = FinancialData(ticker=__params.ticker.upper())
    smodel = SModel(params=__params)
    # smodel.set_params(params=__params)
    # mas = stock.get_moving_average(lags=[__params.lag])
    result = smodel.predict(days=days)
    return result.get_json_result()
    #dao.update_model_status(model_id=get_prediction_model_hash(model_params=__params.get_dict()), status=1)
    #return prediction.get_json_result()

@celery.task()
def build_model(params):
    # only json can be passed to celery task, convert params json to params object
    __params = PredictionParam()
    result = __params.set_params_dict(params=params)
    if result == False:
        return 'Error'
    print('[Predict Task] task received with params\n', __params)
    stock = FinancialData(ticker=__params.ticker.upper())
    smodel = SModel(params=__params)
    ma = stock.get_moving_average(lag=__params.lag, start_date=__params.datetime(), years=__params.training_years)
    if ma['code'] != 1:
        dao.update_model_status(model_id= smodel.model_id, status = -2)
        return ma
    
    model = smodel.create_trained_model(training_set =  ma['data'])
    result = 1 if model != None else 0
    return result

@celery.task()
def evaluate_prediction(params, evaluation_end = None):
    __params = PredictionParam()
    result = __params.set_params(params=params)
    if result == False:
        return 'Error'
    print('[Predict Task] task received with params\n', __params)
    # get the stock data
    stock = FinancialData(ticker=__params.ticker.upper())
    # create the model
    smodel = SModel(params=params)

    if evaluation_end:
        evaluation_end = datetime.datetime.strptime(evaluation_end, '%Y-%m-%d')
    else:
        evaluation_end = stock.max_date
    # evaluate params
    if evaluation_end > stock.max_date:
        evaluation_end = stock.max_date

    days = (evaluation_end - smodel.end_date).days

    # original = stock.get_data(start_date=smodel.start_date,end_date= evaluation_end)
    train = stock.get_moving_average(lag=params.lag, start_date=params.datetime(), years=params.training_years)
    test = stock.get_moving_average(lag=params.lag, start_date=smodel.start_date, end_date=evaluation_end)
    # print(future, original['data'])

    return smodel.evaluate_prediction(days,test,train)


@celery.task()
def migrate_data(start):
    print('LOGS: migrate_data')
    migration = Migration()
    migration.set_setting(key='migration', value=start)
    try:
        inprogress = migration.migrate() == False
        if inprogress:
            return migration.get_current_migration_progress()
    except Exception as ex:
        print(ex)
    finally:
        print('MIGRATION FINISHED')