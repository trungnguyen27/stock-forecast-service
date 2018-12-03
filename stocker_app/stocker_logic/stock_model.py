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
import pickle, os

from stocker_app.config.setting import configs
from stocker_app.utils import database_utils as dbu

model_path = configs['model_path']

class SModel():

    def __init__(self, stock = None):
        stock.describe_stock()
        self.stock = stock
        self.intialize_model_parameters()

    def intialize_model_parameters(self, 
                            seasonalities=['monthly', 'quarterly', 'yearly'],
                            changepoint_prior_scale= 0.05,
                            changepoints = pd.DataFrame(),
                            training_years=10,
                            ):
        self.reset_model_paramaters()

        for seasonality in seasonalities:
            if seasonality == 'daily':
                self.daily_seasonality = True
            elif seasonality == 'weekly':
                self.weekly_seasonality = True
            elif seasonality == 'monthly':
                self.monthly_seasonality = True
            elif seasonality == 'yearly':
                self.yearly_seasonality = True
            elif seasonality == 'quarterly':
                self.quarterly_seasonality = True
    
        if not changepoints.empty:
            self.changepoints = changepoints
        else:
            self.changepoints = pd.DataFrame()
        self.changepoint_prior_scale = changepoint_prior_scale
        self.training_years = training_years
            
        print('Seasonalities: Daily[{}] Weeky[{}] Monthly[{}] Yearly[{}] Quarterly[{}]'
                .format(self.daily_seasonality, 
                    self.weekly_seasonality,
                    self.monthly_seasonality,
                    self.yearly_seasonality,
                    self.quarterly_seasonality))
        print('Changepoints: {}'.format(' '.join(changepoints if not changepoints.empty else ['None'])))
        print('Number of training years: {}'.format(training_years))

    def get_model_params(self, lag):
        params = {
            'ticker': self.stock.ticker,
            'daily_seasonality': self.daily_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'monthly_seasonality': self.monthly_seasonality,
            'yearly_seasonality': self.yearly_seasonality,
            'quarterly_seasonality': self.quarterly_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'training_years': self.training_years,
            'prediction_start': str(self.stock.max_date.date()),
            'lag': lag,
        }
        print('Current Model Params: ', params)
        return params

    def reset_model_paramaters(self):
        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = False
        self.yearly_seasonality = False
        self.quarterly_seasonality = False
        self.changepoints = pd.DataFrame()
        self.training_years = 0
        self.test_months = 0
       
    def build_model(self, evaluation = False):
         # filter changepoints
        if not self.changepoints.empty:
            if evaluation:
                upper_bound = self.stock.max_date - pd.DateOffset(months=self.test_months)
                lower_bound = upper_bound - pd.DateOffset(years=self.training_years)
                changepoints = self.changepoints[((self.changepoints['ds'] < upper_bound) & (self.changepoints['ds'] > lower_bound))]['ds']
            else:
                changepoints = self.changepoints['ds']
        else:
            changepoints = None

        #changepoints = self.changepoints['ds'] if not self.changepoints.empty else None
        model = Prophet(interval_width=0.2, 
                        daily_seasonality=self.daily_seasonality, 
                        weekly_seasonality=self.weekly_seasonality, 
                        yearly_seasonality=self.yearly_seasonality, 
                        changepoint_prior_scale=self.changepoint_prior_scale, 
                        changepoints=changepoints)
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        if self.quarterly_seasonality:
            model.add_seasonality(name='quarterly', period=90, fourier_order=5)
        if self.yearly_seasonality:
            model.add_seasonality(name='yearly', period=365, fourier_order=5)
        
        return model

    def predict(self, training_sets = dict(), lags = [5], days = 30):
        # Use past self.training_years years for training
        if not training_sets:
            training_sets['%s price' %self.stock.ticker] = self.stock.data['Close']
        predictions = dict()
        # Lags that corresponds to moving averages
        self.lags = lags
        
        for i, (key, column) in enumerate(training_sets.items()):
            result = self.predict_single_dataset(train = column, lag=lags[i], days = days)
            predictions[key] = result['predicted']
        self.predictions = predictions
        return predictions

    def predict_single_dataset(self, train, lag, days):
        model = self.get_trained_model(train, lag = lag)
        # Future dataframe with specified number of days to predict
        predicted = model.make_future_dataframe(periods=days, freq='D')
        predicted = model.predict(predicted)
        # Only concerned with future dates
        future = predicted[predicted['ds'] >= self.stock.max_date.date()]
    
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
                print(c_data.ix[:, ['Date', 'Adj. Close', 'delta']][:5])

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

    def get_trained_model(self, train, lag):
        model_hash= dbu.get_prediction_model_hash(model_params = self.get_model_params(lag= lag))
        model_dir = '%s/%s.pkl' %(model_path, model_hash)
        print('Hash', model_hash)

        from stocker_app.stock_database.DAO import DAO
        dao = DAO()

        status = dao.get_model_status(model_id = model_hash)

        model = dao.get_prediction_model(model_id = model_hash)
        if model != None:
            model = pickle.loads(model)
        else:
            model = self.build_model()
            model.fit(train)
            pklobj = pickle.dumps(model)
            dao.save_prediction_model( model_params = self.get_model_params(lag= lag), model = pklobj)
        return model

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

    def evaluate_prediction(self, training_sets = None, test_months=5, start_date=None, end_date=None, nshares = None):
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            self.test_months = test_months
            start_date = self.stock.max_date - pd.DateOffset(months=test_months)
        if end_date is None:
            end_date = self.stock.max_date
        
        path = 'prediction evaluation {}.xlsx'.format(str(datetime.datetime.now()))
        writer = pd.ExcelWriter(path = path)
        export_data = pd.DataFrame()
        for (key, column) in training_sets.items():
             # Training data starts self.training_years years before start date and goes up to start date
            train = column[(column['ds'] < start_date.date()) & 
                            (column['ds'] > (start_date - pd.DateOffset(years=self.training_years)).date())]
    
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

        