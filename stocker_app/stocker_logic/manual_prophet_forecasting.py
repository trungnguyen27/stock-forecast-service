from stocker_app.stocker_logic.stock_model import SModel
from stocker_app.stock_database.financial_data import FinancialData
from stocker_app.config import configs
import pandas as pd

stock = FinancialData(ticker="VNM")
stock.describe_stock()

stock.plot_stock()

lags = [1, 30]
days = 7

mas = stock.get_moving_average(lags = lags, columns=['close'])

stock.plot_stock(show_data = False, show_volume=True, moving_averages=mas)

smodel = SModel(stock=stock)

smodel.predict(training_sets=mas,days = 90)
smodel.plot_predictions()
smodel.plot_detail_prediction()

changepoints = smodel.changepoint_date_analysis(training_sets=mas)
print(changepoints)
smodel.intialize_model_parameters(changepoints = changepoints, changepoint_prior_scale=0.5)
smodel.evaluate_prediction(mas)

# smodel.predict(mas, days= days)
# smodel.plot_predictions()
# smodel.intialize_model_parameters(changepoint_prior_scale=0.1)
# mas['actual'] = stock.get_data()
# smodel.evaluate_prediction(training_sets=mas)


# smodel.set_moving_averages(lags=[90], columns=['close'])
# #smodel.plot_stock(show_data=False, show_moving_avg=True, show_volume=True)
# #smodel.predict(use_moving_avg=True, days = 90)
# smodel.evaluate_prediction()
# smodel.plot_stock(show_moving_avg=True)
#smodel.plot_stock(show_moving_avg=True, columns=['close'])
#smodel.build_model()

#smodel.predict(use_moving_avg=True,days=90,training_years=10)

#smodel.changepoint_prior_analysis(changepoint_priors=[0.05, 0.1, 0.5])

#smodel.evaluate_prediction()

#smodel.plot_predicted_range()
#smodel.plot_history_and_prediction()

# future = m.make_future_dataframe(periods=periods)
# future.tail()

# forecast = m.predict(future)
# history_segmnet = forecast[:-periods]
# forecast_segment = forecast[-periods:]
# print(history_segmnet)
# forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail()

# # fig2 = m.plot_components(forecast)
# # fig2.show()

# # plt.plot(forecast.tail()['ds'], forecast.tail()['yhat'])
# # plt.plot(forecast['ds'], forecast['yhat'])
# # plt.show()
# # print(df.index)

# fig = plt.figure(figsize=(12,8))

# first_plot_segment = history_segmnet.tail()
# second_plot_segment= forecast_segment.tail()
# uncertainty_segment = forecast[:,-periods]

# #plt.plot(first_plot_segment['ds'], first_plot_segment['yhat'], color='darkgray')
# plt.plot(second_plot_segment['ds'], second_plot_segment['yhat'], color='powderblue')
# plt.legend(loc='best')
# plt.title('Prediction')
# # Plot the uncertainty interval as ribbon
# plt.fill_between(uncertainty_segment['ds'].dt.to_pydatetime(), uncertainty_segment['yhat_upper'], uncertainty_segment['yhat_lower'], alpha = 0.3, 
#                        facecolor = 'g', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')
# plt.show()
# print(forecast_segment[['ds','yhat']])

