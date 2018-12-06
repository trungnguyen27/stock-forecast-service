import pandas as pd
class Prediction():
    def __init__(self, params, status = 0, prediction = pd.DataFrame()):
        self.params = params
        self.status = status
        self.prediction = prediction

    def get_json_result(self):
        self.params = self.params.get_dict()
        self.prediction = self.prediction[['ds', 'direction', 'y', 'yhat_upper', 'yhat_lower']].to_dict('records')
        return self.__dict__