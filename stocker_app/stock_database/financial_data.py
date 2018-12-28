import pandas as pd
import numpy as np
import warnings
import datetime
from sklearn.preprocessing import MinMaxScaler
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
from stocker_app.config import configs
from stocker_app.stock_database.DAO import DAO
dao = DAO()

warnings.filterwarnings('ignore')

csv_path = configs['csv_path']
metastock_name = configs['metastock_name']

class FinancialData():
    currency = '000VND'
    def __init__(self, ticker = "VIC"):
        #["Ticker","Date","OpenFixed","HighFixed","LowFixed","CloseFixed","Volume","Open","High","Low","Close","VolumeDeal","VolumeFB","VolumeFS"]
        self.ticker = ticker.capitalize()
        try:

            data = dao.get_data(ticker=ticker)
            print(data.head())
            if data.empty == True:
                self.data = pd.DataFrame()
                return
            # data = data[(data['Ticker']== ticker)]
            self.columns = ["id", "ticker","date","open", "high", "low", "close", "volume"]
            data.columns= self.columns
            data['date']=pd.to_datetime(data.date)

            data.index = data['date']
            data = data.sort_values(by=['date'], ascending=[True])
            #remove duplicates
            data = data.drop_duplicates(subset=['ticker', 'date'])

            data = data.resample('D').fillna(method='ffill')

            data['ds']=data.index
            data['y']=data['close']

            if ('adjclose' not in data.columns):
                data['adjclose'] = data['close']
                data['adjopen'] = data['open']
            
            self.data = data
            # Minimum and maximum date in range
            self.min_date = min(data['date'])
            self.max_date = max(data['date'])

            self.years = (self.max_date - self.min_date).days/365
            
            self.max_price = np.max(self.data['y'])
            self.min_price = np.min(self.data['y'])
            
            self.min_price_date = self.data[self.data['y'] == self.min_price]['date']
            self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
            self.max_price_date = self.data[self.data['y'] == self.max_price]['date']
            self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
            
            # The starting price (starting with the opening price)
            self.starting_price = float(self.data['adjopen'].iloc[0])
            
            # The most recent price
            self.most_recent_price = float(self.data['y'].iloc[-1])
        except Exception as ex:
            print(ex)

    def get_data_description(self):
        dictObj =  self.__dict__
        dictObj['data'] = dictObj['data'].to_dict('records')
        return dictObj

    # Get the data from start date until the last record
    def get_data(self, start_date = None, end_date = None):
        if self.data.empty:
            print ('Stock data is empty')
            return pd.DataFrame()
        else:
            print('Stock History Data of %s' %(self.ticker))
            start = self.min_date
            end = self.max_date
            if start_date:
                start = start_date
            if end_date:
                end = end_date
            
            if end < start:
                return []
            
            result = self.data[((self.data['ds'] >= start) & (self.data['ds'] <= end))]
            return result.dropna()

    def describe_stock(self):
        print(self.data.head())
        print('%d years of %s stock history data \n[%s To %s]' %(self.years, self.ticker, self.min_date, self.max_date))
        print('Lowest price on: %s with %d %s\nHighest price on: %s with %d %s' %(self.min_price_date, self.min_price, self.currency, self.max_price_date, self.max_price, self.currency))


    def get_moving_average_depreated(self, lag = [5], columns=['close'], start_date=None):
        filtered_data= self.data.copy()
        if start_date:
            filtered_data = filtered_data[(self.data['ds'] > start_date)]
        # with each columns, we calculate the moving average with lag
        for column in columns:
            try:
                for i, lag in enumerate(lag):
                    data = filtered_data[['ds', column]]
                    data = data.rename(columns = {column: 'y'})
                    data['y'] = data['y'].rolling(lag).mean().round(2)
                    data = data.dropna()
                    moving_averages['MA_%d_%s' %(lag, column)] = data
            except Exception as e:
                print('An error occured:')
                print(e)

        self.lag = lag
        self.moving_averages = moving_averages

        lag_str =""
        for lag in lag:
            lag_str += str(lag) + ' '
        print('Moving averages generated: [{}] on columns [{}]'.format(lag_str, '-'.join(columns)))

        return moving_averages

    def get_moving_average(self, lag = 5, column='close', start_date=None, end_date=None, years = None):
        filtered_data= self.data.copy()
        startD = self.min_date
        endD = self.max_date
        year_span = self.years
        if start_date:
            startD = start_date
            if startD > self.max_date or startD < self.min_date:
                return {
                    'code': -1,
                    'message': 'Start date must be within the timespan'
                }
        
        if end_date:
            endD = end_date
            if endD < startD:
                return {
                    'code': -1,
                    'message': 'Start date must be within the timespan'
                }

        if years and not end_date:
            endD = startD + datetime.timedelta(days = years * 365)
            if endD > self.max_date:
                return {
                    'code': -2,
                    'message': 'Exceed the maximum date time'
                }

        year_span = (endD- startD).days/30

        filtered_data = filtered_data[((self.data['ds'] > startD) & (self.data['ds'] < endD))]
        # with each columns, we calculate the moving average with lag
        data = filtered_data[['ds', column]]
        data = data.rename(columns = {column: 'y'})
        data['y'] = data['y'].rolling(lag).mean().round(2)
        data = data.dropna()

        self.lag = lag

        return {
            'code': 1,
            'ticker': self.ticker,
            'data':data,
            'params':{
                'lag': lag,
                'label': column,
                'start_date': str(startD.date()),
                'end_date': str(endD.date()),
                'years': year_span
            }
        }

    def plot_stock(self, columns=['close'], show_data=True, show_volume=False, moving_averages = None):
        self.reset_plot()
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        plt.style.use('seaborn')
        
        if show_data:
            for i, column in enumerate(columns):
                plt.plot(self.data['ds'], self.data[column], color=colors[i])

        if show_volume:
            monthly_resampled_volume=self.data[['ds', 'Volume']].resample('M', on='ds').sum()
            min_max_scaler = MinMaxScaler()
            scaled_volume = min_max_scaler.fit_transform(np.array(monthly_resampled_volume).reshape(-1,1))
            scaled_volume *= max(self.data['close'])/2
            plt.bar(monthly_resampled_volume.index, scaled_volume.flatten(), width=10)
        
        if moving_averages:
            for (col, avg) in moving_averages.items():
                plt.plot(avg['ds'], avg['y'] , ls='--', label = col)
                
        #plt.plot(self.data['ds'], moving_avg, color='powderblue')
        plt.title('{} on {}'.format(self.ticker, ' '.join(columns)))
        plt.legend(loc='best')
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