class PredictionParam():
    def __init__(
        self,
        ticker = 'VIC',
        daily_seasonality = False,
        weekly_seasonality = False,
        monthly_seasonality = False,
        yearly_seasonality=  False,
        quarterly_seasonality=  False,
        changepoint_prior_scale = 0.05,
        training_years = 10,
        date = '2010-01-01',
        lag = 5
        ):
        self.ticker = ticker
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.monthly_seasonality = monthly_seasonality
        self.yearly_seasonality=  yearly_seasonality
        self.quarterly_seasonality=  quarterly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.training_years = training_years
        self.date = date
        self.lag = lag
    def get_dict(self):
        return self.__dict__

params = PredictionParam()
print(params.get_dict())