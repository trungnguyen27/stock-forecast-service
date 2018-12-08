from stocker_app.utils.database_utils import get_prediction_model_hash
class PredictionParam():
    def __init__(
        self,
        ticker = 'VIC',
        seasonalities='m-q-y',
        changepoint_prior_scale = 0.05,
        training_years = 5,
        date = '2010-01-01',
        lag = 5
        ):
        
        self.ticker = ticker.upper()
        self.__parse_seasonalities__(seasonalities_str=seasonalities)
        self.changepoint_prior_scale = changepoint_prior_scale
        self.training_years = training_years
        self.date = date
        self.lag = lag
    
    def __parse_seasonalities__(self, seasonalities_str):
        seasonalities = seasonalities_str.split('-')
        self.daily_seasonality = False
        self.weekly_seasonality = False
        self.monthly_seasonality = False
        self.yearly_seasonality = False
        self.quarterly_seasonality = False
        for s in seasonalities:
            if s == 'd':
                self.daily_seasonality = True
            elif s =='w':
                self.weekly_seasonality = True
            elif s =='m':
                self.monthly_seasonality = True
            elif s == 'y':
                self.yearly_seasonality=  True
            elif s == 'q':
                self.quarterly_seasonality=  True


    def set_params(self, params):
        self.ticker = params.ticker
        self.changepoint_prior_scale = params.changepoint_prior_scale
        self.training_years = params.training_years
        self.date = str(params.prediction_start)
        self.lag = params.lag
        self.daily_seasonality = params.daily_seasonality
        self.weekly_seasonality = params.weekly_seasonality
        self.monthly_seasonality = params.monthly_seasonality
        self.yearly_seasonality = params.yearly_seasonality
        self.quarterly_seasonality = params.quarterly_seasonality

    def set_params_dict(self, params):
        if type(params) is dict:
            self.ticker = params['ticker']
            self.changepoint_prior_scale = params['changepoint_prior_scale']
            self.training_years = params['training_years']
            self.date = str(params['date'])
            self.lag = params['lag']
            self.daily_seasonality = params['daily_seasonality']
            self.weekly_seasonality = params['weekly_seasonality']
            self.monthly_seasonality = params['monthly_seasonality']
            self.yearly_seasonality = params['yearly_seasonality']
            self.quarterly_seasonality = params['quarterly_seasonality']
            return True
        return False

    def get_description(self):
        ticker = 'Ticker: %s\n' %self.ticker
        seasonality = 'Seasonalities: Daily[{}] Weeky[{}] Monthly[{}] Yearly[{}] Quarterly[{}]\n'.format(
                        self.daily_seasonality, 
                        self.weekly_seasonality,
                        self.monthly_seasonality,
                        self.yearly_seasonality,
                        self.quarterly_seasonality)
        training_years = 'Number of training years: {}\n'.format(self.training_years)
        lag = 'Lag: %d\n' %self.lag
        date = 'Prediction start at: %s\n' %self.date
        desc =ticker + lag + seasonality + training_years + date
        return desc

    def parse_params_from_request(self, ticker, lag, piror, seasonalities, date):
        self.lag = int(lag)

    
    def get_dict(self):
        return self.__dict__

    def get_hash(self):
        self.get_description()
        return get_prediction_model_hash(self.__dict__)