from stocker_app.stock_database.migration.parse_csv import Migration
from stocker_app.stock_database.DAO import DAO
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.stocker_logic.stock_model import SModel


ticker = 'VNM'
lags = [10]
prior = 0.1
stock = FinancialData(ticker)
smodel = SModel(stock)
mas = stock.get_moving_averages(lags=lags)
smodel.intialize_model_parameters(changepoint_prior_scale=prior)
predictions = smodel.predict(training_sets = mas, lags =lags,  days = 30)


def write(df, ticker, start_date, prior, lag ):
    dao = DAO()
    data = dao.save_prediction(ticker = ticker, start_date = start_date, prior = prior, lag = lag, prediction_df = df)
    print(data)

for index, (key, result) in enumerate(predictions.items()):
    trimmed = result[['ds', 'yhat', 'y', 'yhat_upper', 'yhat_lower']]
    result.describe()
    print(trimmed.head())