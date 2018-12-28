import pandas as pd
class Prediction():
    def __init__(self, params, status = 0, interval_width=0, days=30, prediction = pd.DataFrame(), past = pd.DataFrame(), changepoints= []):
        self.params = params
        self.status = status
        self.prediction = prediction
        self.past = past
        self.changepoints = changepoints
        self.days = days
        self.interval_width = 0

    def get_json_result(self):
        self.params = self.params.get_dict()
        self.prediction = self.prediction[['ds', 'direction', 'y', 'yhat_upper', 'yhat_lower']].to_dict('records')
        self.past = self.past[['ds','y']].to_dict('records')
        self.changepoints = self.changepoints.to_dict()
        return self.__dict__