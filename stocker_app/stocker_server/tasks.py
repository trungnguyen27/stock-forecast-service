from stocker_app.stocker_logic.stock_model import SModel
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stock_database.migration.parse_csv import Migration
from stocker_app.models import PredictionParam
from flask import jsonify
from stocker_app import celery

@celery.task()
def predict(params):
    # only json can be passed to celery task, convert params json to params object
    __params = PredictionParam()
    result = __params.set_params_dict(params=params)
    if result == False:
        return 'Error'
    print('[Predict Task] task received with params\n', __params)
    stock = FinancialData(ticker=__params.ticker.upper())
    smodel = SModel(stock)
    smodel.set_params(params=__params)
    mas = stock.get_moving_averages(lags=[__params.lag])
    prediction = smodel.predict(training_set =  list(mas.values())[0], days = 30)
    return prediction.get_json_result()

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