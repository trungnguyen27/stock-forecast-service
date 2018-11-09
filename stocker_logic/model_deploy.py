from global_configs import configs
import pickle
from stocker_logic.stock_model import SModel
from stock_database.financial_data import FinancialData

model_path = configs['model_path']

stock = FinancialData(ticker="VIC")
stock.describe_stock()

lags = [1]
days = 7

mas = stock.get_moving_averages(lags = lags, columns=['Close'])

smodel = SModel(stock=stock)

for (key, ma) in mas.items():
    model = smodel.get_trained_model(ma)
    #serializing our model to a file called model.pkl
    pickle.dump(model, open("%s/prophet_model_%s.pkl" %(model_path,key),"wb"))

# #loading a model from a file called model.pkl
# model = pickle.load(open("prophet_model_MA_1_Close.pkl","b"))