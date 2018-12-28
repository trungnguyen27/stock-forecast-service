import pandas as pd
import numpy as np
from fbprophet import Prophet
import pytrends
from pytrends.request import TrendReq
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle, os, asyncio

from stocker_app.config import configs
from stocker_app.models import Prediction, PredictionParam

from stocker_app.stock_database.DAO import DAO
dao = DAO()

model_path = configs['model_path']

class SModel():

    def __init__(self, params = None):
        #stock.describe_stock()
        # if stock != None:
        #     self.stock = stock
        if params == None:
            self.intialize_model_parameters()
        else:
            self.set_params(params)
        self.initialize_model_variables()

    def set_params(self, params):
        self.params = params
        self.changepoints = pd.DataFrame()

    def intialize_model_parameters(self,
                            lag = 5,
                            ticker = 'VIC',
                            label = 'close',
                            seasonalities='m-q-y',
                            changepoint_prior_scale= 0.05,
                            changepoints = pd.DataFrame(),
                            training_years=5,
                            date='2010-01-01'
                            ):
        self.params = PredictionParam(
                                    ticker = ticker,
                                    label = label,
                                    lag = lag, 
                                    seasonalities=seasonalities, 
                                    changepoint_prior_scale=changepoint_prior_scale,
                                    training_years = training_years,
                                    date=date)
        if not changepoints.empty:
            self.changepoints = changepoints
        else:
            self.changepoints = pd.DataFrame()
    
    def initialize_model_variables(self):
           # model_id, status, prediction
        self.model_id = self.params.get_hash()
        self.status = self.get_model_status()
        self.prediction = Prediction(status= self.status, params = self.params)

        self.start_date = datetime.datetime.strptime(self.params.date, '%Y-%m-%d')
        self.end_date = self.start_date + datetime.timedelta(days = 365 * self.params.training_years)
        print('[Default]: \n%s', self.params.get_description())

    def get_model_params(self):
        print('Current Model Params: ', self.params)
        return self.params

    def get_model_status(self):
        try:
            status = dao.get_model_status(model_id=self.model_id)
            return status
        except Exception as ex:
            print(ex)
        return False

    def reset_model_paramaters(self):
        self.params = PredictionParam()
       
    def build_model(self, evaluation = False):
         # filter changepoints
        if not self.changepoints.empty:
            if evaluation:
                upper_bound = self.stock.max_date - pd.DateOffset(months=self.test_months)
                lower_bound = upper_bound - pd.DateOffset(years=self.params.training_years)
                changepoints = self.changepoints[((self.changepoints['ds'] < upper_bound) & (self.changepoints['ds'] > lower_bound))]['ds']
            else:
                changepoints = self.changepoints['ds']
        else:
            changepoints = None

        #changepoints = self.changepoints['ds'] if not self.changepoints.empty else None
        model = Prophet(interval_width=0.2, 
                        daily_seasonality=self.params.daily_seasonality, 
                        weekly_seasonality=self.params.weekly_seasonality, 
                        yearly_seasonality=self.params.yearly_seasonality, 
                        changepoint_prior_scale=self.params.changepoint_prior_scale, 
                        changepoints=changepoints)
        if self.params.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        if self.params.quarterly_seasonality:
            model.add_seasonality(name='quarterly', period=90, fourier_order=5)
        if self.params.yearly_seasonality:
            model.add_seasonality(name='yearly', period=365, fourier_order=5)
        
        return model

    def predict_deprecated(self, training_set = pd.DataFrame(), days = 30):
        # Use past self.training_years years for training
        if training_set.empty == True:
            print('[Prediction] No training set found!')
            return self.prediction

        result = self.predict_single_dataset(training_set = training_set, days = days)
        self.prediction.prediction= result['predicted']
        return self.prediction

    def retreive_param_info(model_id):
        params_db = dao.get_prediction_model(model_id = model_id)
        params = PredictionParam()
        params.set_params(params=params_db)
        self.prediction = Prediction(status= this.get_, params = self.params)

    def predict(self,  days = 30):
         # Use past self.training_years years for training
        # if training_set.empty == True:
        #     print('[Prediction] No training set found!')
        #     return self.prediction
        model_id = self.model_id
        status = dao.get_model_status(model_id)
        if status['status'] != 1:
            return {
                'message': 'model is not ready',
                'code': -1
            }

        result = self.predict_single_dataset(model_id, days = days)
        self.prediction.prediction= result['prediction']
        self.prediction.past = result['predicted']
        self.prediction.changepoints = result['changepoints']
        self.prediction.days = days
        self.interval_width = result['interval_width']
        return self.prediction

    def predict_single_dataset_deprecated(self, model_id, days):
        model =  dao.get_prediction_model(model_id)
        model = pickle.loads(model.model_pkl)
        # Future dataframe with specified number of days to predict
        predicted = model.make_future_dataframe(periods=days, freq='D')
        predicted = model.predict(predicted)
        # Only concerned with future dates
        future = predicted[predicted['ds'] >= datetime.strptime(self.params.date)]
        # self.stock.max_date.date()
        # Remove the weekends
        #future = self.remove_weekends(future)

        future = future.dropna()
        
        # Calculate whether increase or not
        predicted['diff'] = predicted['yhat'].diff()
        future['diff'] = future['yhat'].diff()
    
        predicted = predicted.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        predicted['direction'] = (predicted['diff']> 0) *1
        predicted['y'] = predicted['yhat']
        result = dict()
        result['predicted'] = predicted 
        result['future'] = future
        
        return result


    def predict_single_dataset(self, model_id, days):
        model =  dao.get_prediction_model(model_id = model_id)
        model = pickle.loads(model.model_pkl)
        # Future dataframe with specified number of days to predict
        predicted = model.make_future_dataframe(periods=days, freq='D')
        predicted = model.predict(predicted)

        predicted = predicted.dropna()
        print('NOW PARSING DATE: ',self.params.date)
        future = predicted[predicted['ds'] >= datetime.datetime.strptime(self.params.date, "%Y-%m-%d")]
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff']> 0) *1
        future['y'] = future['yhat']
        predicted['y'] = predicted['yhat']
        
        return {
            'prediction': future,
            'predicted': predicted,
            'changepoints': model.changepoints,
            'interval_width': model.interval_width
        }

    def create_trained_model(self, training_set):
        model = dao.get_prediction_model(model_id = self.model_id)
        if model != None:
            model = pickle.loads(model.model_pkl)
        else:
            success = dao.update_model_status(model_id=self.model_id, status=2)
            if success == False:
                print('[Get train model] Error while updateing model status')
                return None
            model = self.build_model()
            model.fit(training_set)
            pklobj = pickle.dumps(model)
            print('PM PARAMS TO SAVE', self.get_model_params())
            saved =  dao.save_prediction_model( model_params = self.get_model_params(), model = pklobj)
            if saved == False:
                return None
        return model
        
    def retrieve_google_trends(self, search, date_range):
        # Set up the trend fetching object
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [search]

        try:
        
            # Create the search object
            pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='news')
            
            # Retrieve the interest over time
            trends = pytrends.interest_over_time()

            related_queries = pytrends.related_queries()

        except Exception as e:
            print('\nGoogle Search Trend retrieval failed.')
            print(e)
            return
        
        return trends, related_queries
        
    def changepoint_date_analysis(self, training_sets = None, search=None):
        self.reset_plot()

        model = self.build_model()

        for (key, column) in training_sets.items():
            # Use past self.training_years years of data

            train = column[column['ds'] > (self.stock.max_date - pd.DateOffset(years=int(self.stock.years))).date()]
            model.fit(train)
            
            # Predictions of the training data (no future periods)
            future = model.make_future_dataframe(periods=0, freq='D')
            future = model.predict(future)
        
            train = pd.merge(train, future[['ds', 'yhat']], on = 'ds', how = 'inner')
            
            changepoints = model.changepoints

            train = train.reset_index(drop=True)
            change_indices = []
            for changepoint in (changepoints):
                change_indices.append(train[train['ds'] == changepoint.date()].index[0])
            
            c_data = train.ix[change_indices, :]
            deltas = model.params['delta'][0]
            
            c_data['delta'] = deltas
            c_data['abs_delta'] = abs(c_data['delta'])
            
            # Sort the values by maximum change
            c_data = c_data.sort_values(by='abs_delta', ascending=False)

            # Limit to 10 largest changepoints
            c_data = c_data[:10]
            # Changepoints and data
            if not search:
            
                print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
                print(c_data.ix[:, ['Date', 'adjclose', 'delta']][:5])

                # Line plot showing actual values, estimated values, and changepoints
                self.reset_plot()
                
                # Set up line plot 
                plt.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Stock Price')
                plt.plot(future['ds'], future['yhat'], color = 'navy', linewidth = 2.0, label = 'Modeled')

                self.plot_changepoints(changepoints= c_data, train=train)

                plt.legend(prop={'size':10}) 
                plt.xlabel('Date')  
                plt.ylabel('Price (VNĐ)')  
                plt.title('Stock Price with Changepoints')
            
            # Search for search term in google news
            # Show related queries, rising related queries
            # Graph changepoints, search frequency, stock price
            if search:
                date_range = ['%s %s' % (str(min(train['Date']).date()), str(max(train['Date']).date()))]

                # Get the Google Trends for specified terms and join to training dataframe
                trends, related_queries = self.retrieve_google_trends(search, date_range)

                if (trends is None)  or (related_queries is None):
                    print('No search trends found for %s' % search)
                    return

                print('\n Top Related Queries: \n')
                print(related_queries[search]['top'].head())

                print('\n Rising Related Queries: \n')
                print(related_queries[search]['rising'].head())

                # Upsample the data for joining with training data
                trends = trends.resample('D').sum()

                trends = trends.reset_index(level=0)
                trends = trends.rename(columns={'date': 'ds', search: 'freq'})

                # Interpolate the frequency
                trends['freq'] = trends['freq'].interpolate()

                # Merge with the training data
                train = pd.merge(train, trends, on = 'ds', how = 'inner')

                # Normalize values
                train['y_norm'] = train['y'] / max(train['y'])
                train['freq_norm'] = train['freq'] / max(train['freq'])
                
                self.reset_plot()

                # Plot the normalized stock price and normalize search frequency
                plt.plot(train['ds'], train['y_norm'], 'k-', label = 'Stock Price')
                plt.plot(train['ds'], train['freq_norm'], color='goldenrod', label = 'Search Frequency')

                self.plot_changepoints(changepoints= c_data, train = train)

                # Plot formatting
                plt.legend(prop={'size': 10})
                plt.xlabel('Date')  
                plt.ylabel('Normalized Values')  
                plt.title('%s Stock Price and Search Frequency for %s' % (self.stock.ticker, search))
        plt.show()
        self.changepoints = c_data
        return c_data

    def export_results(self, name="Prediction"):
        if not self.predictions:
            print("Predictions not found")
            return
        try:
            writer = pd.ExcelWriter(path='%s.xlsx' %name)
            for (key, result) in self.predictions.items():
                content = result[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'direction']]
                content = content.rename(columns={'ds': 'Date', 'yhat': 'Estimate', 'yhat_lower': 'Low', 'yhat_upper': 'High'})
                content.to_excel(excel_writer=writer, header=name + str(datetime.datetime.now()),sheet_name = key )
            writer.save()
            print('%s exported successfully' %name)
        except Exception as ex:
            print('Error while exporting!')
            print(ex)
        

    def plot_predictions(self):
        self.reset_plot()
        plt.title('Predicted Price on Close Price and Moving Averages on {}'.format(self.stock.ticker))
        # lags = self.moving_averages['lags']

        for (key, future) in self.predictions.items():
            history_range = future[future['ds'] < self.stock.max_date]
            future_range = future[future['ds'] > self.stock.max_date]
            plt.plot(history_range['ds'], history_range['yhat'], label=key)
            plt.plot(future_range['ds'], future_range['yhat'], label='%s predicted' %key)
            # Plot the uncertainty interval
            plt.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'],
                                future['yhat_lower'],
                                alpha = 0.3, edgecolor = 'k', linewidth = 0.6)
            self.plot_changepoints(train=history_range)
              

        plt.legend(loc='best')
        plt.show()

    def plot_detail_prediction(self):
        self.reset_plot()
        # Set up plot
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12

        for (key, prediction) in self.predictions.items():
            future = prediction[prediction['ds'] > self.stock.max_date]
             # Plot the predictions and indicate if increase or decrease
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            future_increase = future[future['direction'] == 1]
            future_decrease = future[future['direction'] == 0]

            # Plot the estimates
            ax.plot(future_increase['ds'], future_increase['yhat'], 'g^', ms = 12, label = 'Pred. Increase')
            ax.plot(future_decrease['ds'], future_decrease['yhat'], 'rv', ms = 12, label = 'Pred. Decrease')

            # Plot errorbars
            ax.errorbar(future['ds'].dt.to_pydatetime(), future['yhat'], 
                        yerr = future['yhat_upper'] - future['yhat_lower'], 
                        capthick=1.4, color = 'k',linewidth = 2,
                    ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')
       
        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10}) 
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (%s)' %self.stock.currency) 
        plt.xlabel('Date')  
        plt.title('Predictions for %s' % self.stock.ticker) 
        plt.show()


    def evaluate_prediction_deprecated(self, training_set = None, test_months=5, start_date=None, end_date=None, nshares = None):
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            self.test_months = test_months
            start_date = self.stock.max_date - pd.DateOffset(months=test_months)
        if end_date is None:
            end_date = self.stock.max_date
        
         # Training data starts self.training_years years before start date and goes up to start date
        train = column[(column['ds'] < start_date.date()) & 
                        (column['ds'] > (start_date - pd.DateOffset(years=self.params.training_years)).date())]

        # Testing data is specified in the range
        test = column[(column['ds'] >= start_date.date()) & (column['ds'] <= end_date.date())]

        model = self.build_model(evaluation=True)
        model.fit(train)

        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 150, freq='D')
        future = model.predict(future)

        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')

        train = pd.merge(train, future, on = 'ds', how = 'inner')

        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()
        
        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test.ix[i, 'y'] < test.ix[i, 'yhat_upper']) & (test.ix[i, 'y'] > test.ix[i, 'yhat_lower']):
                test.ix[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])
        last_predicted_price = future.ix[len(future) - 1, 'yhat']*1000
        last_actual_price = test.ix[len(test) - 1, 'y']*1000

        export_data[key] = [train_mean_error, test_mean_error, last_actual_price, last_predicted_price]
        if not nshares:
            # Date range of predictions
            print('\nPrediction Range: {} to {}.'.format(start_date.date(),
                end_date.date()))

            # Final prediction vs actual value
            print('\nPredicted price on {} = {:.2f}VNĐ.'.format(max(future['ds']).date(), last_predicted_price))
            print('Actual price on    {} = {:.2f}VNĐ.\n'.format(max(test['ds']).date(), last_actual_price))

            print('Average Absolute Error on Training Data = {:.2f}VND.'.format(train_mean_error*1000))
            print('Average Absolute Error on Testing  Data = {:.2f}VND.\n'.format(test_mean_error*1000))

            # Direction accuracy
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))


            # Reset the plot
            self.reset_plot()
            
            # Set up the plot
            fig, ax = plt.subplots(1, 1)

            # Plot the actual values
            ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            
            # Plot the predicted values
            ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

            # Plot the uncertainty interval as ribbon
            ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                        facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

            # Put a vertical line at the start of predictions
            plt.vlines(x=min(test['ds']).date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                    linestyles='dashed', label = 'Prediction Start')

            # Plot formatting
            plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price Thousand VNĐ');
            plt.grid(linewidth=0.6, alpha = 0.6)
                    
            plt.title('{} Model Evaluation from {} to {}.'.format(self.stock.ticker,
                start_date.date(), end_date.date()))
        
        export_data = export_data.rename(index={0: 'train mean error', 1: 'test mean error', 2: 'last actual price', 3: 'last predicted price'})
        export_data.to_excel(writer, '%s evaluation' %key, header='evaluation')
        writer.save()
        print('%s exported' %path)
            
        plt.show()

    def evaluate_prediction(self, days, test, train, nshares=None):
        result = self.predict(days=days)
        # two dataset to compare
        future = result.prediction[['ds','yhat','yhat_upper', 'yhat_lower']]
        
        ## evaluation section
        test = pd.merge(test['data'][['ds','y']], future, on = 'ds', how = 'inner')
        train = pd.merge(train['data'][['ds','y']], future, on = 'ds', how = 'inner')

        print(test.head(), train.head())

        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test['y'].iloc[i] < test['yhat_upper'].iloc[i]) & (test['y'].iloc[i] > test['yhat_lower'].iloc[i]):
                test['in_range'].iloc[i] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])
        last_predicted_price = future['yhat'].iloc[len(future) - 1]*1000
        last_actual_price = test['y'].iloc[-1]*1000

        if not nshares:
            # Date range of predictions
            print('\nPrediction Range: {} to {}.'.format(self.start_date.date(),
                self.end_date.date()))

            # Final prediction vs actual value
            print('\nPredicted price on {} = {:.2f}VNĐ.'.format(max(future['ds']).date(), last_predicted_price))
            print('Actual price on    {} = {:.2f}VNĐ.\n'.format(max(test['ds']).date(), last_actual_price))

            print('Average Absolute Error on Training Data = {:.2f}VND.'.format(train_mean_error*1000))
            print('Average Absolute Error on Testing  Data = {:.2f}VND.\n'.format(test_mean_error*1000))

            # Direction accuracy
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * result.interval_width), in_range_accuracy))

        return {
            'model_id': self.model_id,
            'evaluation': {
                'start_date': str(self.start_date.date()),
                'end_date': str(self.end_date.date()),
                'last_predicted_price': last_predicted_price,
                'last_actual_price': last_actual_price,
                'train_me': train_mean_error * 1000,
                'test_me': test_mean_error * 1000,
                'increase_acc': increase_accuracy,
                'decrease_acc': decrease_accuracy,
                'in_range_acc': in_range_accuracy
            },
            'test': test.to_dict('records'),
            'train': train.to_dict('records')
           
        }

    # Reset the plotting parameters to clear style formatting
    # Not sure if this should be a static method
    @staticmethod
    def reset_plot():
        # Restore default parameters
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        
        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'

    def plot_changepoints(self, changepoints = pd.DataFrame(), train = None):
        if changepoints.empty:
            changepoints = self.changepoints
        # Create dataframe of only changepoints
        # Separate into negative and positive changepoints
        cpos_data = changepoints[changepoints['delta'] > 0]
        cneg_data = changepoints[changepoints['delta'] < 0]

         # Changepoints as vertical lines
        plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
                linestyles='dashed', color = 'r', 
                linewidth= 1.2, label='Negative Changepoints')

        plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
                linestyles='dashed', color = 'darkgreen', 
                linewidth= 1.2, label='Positive Changepoints')

        