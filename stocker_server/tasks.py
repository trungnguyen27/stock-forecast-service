from stocker_logic.stock_model import SModel
from stock_database.financial_data import FinancialData
from stocker_server.celery_init import make_celery
from app import app
from flask import jsonify

celery_app = make_celery(app)

@celery_app.task
def predict(ticker, lags, start_date):
    stock = FinancialData(ticker)
    smodel = SModel(stock)
    mas = stock.get_moving_averages(lags=lags)
    predictions = smodel.predict(training_sets = mas, days = 30)
    json=dict()
    for (key, result) in predictions.items():
        trimmed = result[['ds', 'direction', 'y', 'yhat_upper', 'yhat_lower']]
        json[key]=trimmed.to_dict('records')
    return json