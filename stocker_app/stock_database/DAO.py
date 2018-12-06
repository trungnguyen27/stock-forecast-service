import pandas as pd
from stocker_app.stock_database.schemas import database, Price_History, PredictionModel, ModelStatus, App_Setting
from stocker_app.utils import database_utils as dbu
from time import time
class DAO():
    def __init__(self):
        try:
            database.create_tables()
            self.session = database.get_session()
        except Exception as ex:
            print('failed to get session', ex)
        finally:
            print('DAO intialized')

    def set_setting(self, key, value):
        m_setting = self.session.query(App_Setting).filter(App_Setting.key == key).first()
        if m_setting != None:
            m_setting.value = value
        else:
            self.session.add(App_Setting(**{
               'key': 'migration',
               'value': value
            }))
        self.session.commit()
    
    def get_setting(self, key):
        m_setting = self.session.query(App_Setting).filter(App_Setting.key == key).first()
        if m_setting != None:
            return m_setting.value
        else:
            self.session.add(App_Setting(**{
               'key': key,
               'value': -1
            }))
            self.session.commit()
            return -1

    def convert_prediction_model(self, modelobj):
        return {
            'hash_id': modelobj.id,
            'start_date': modelobj.start_date,
            'ticker': modelobj.ticker,
            'prior': modelobj.prior,
            'ma':modelobj.ma
        }

    def save_prediction_model(self, model_params, model):
        try:
            hash_id = model_params.get_hash()
            record = PredictionModel(**{
                'model_id': hash_id,
                'ticker': model_params.ticker,
                'prediction_start': model_params.date,
                'changepoint_prior_scale': model_params.changepoint_prior_scale,
                'lag': model_params.lag,
                'model_pkl': model,
                'daily_seasonality':model_params.daily_seasonality,
                'weekly_seasonality':model_params.weekly_seasonality,
                'monthly_seasonality':model_params.monthly_seasonality,
                'yearly_seasonality':model_params.yearly_seasonality,
                'quarterly_seasonality':model_params.quarterly_seasonality,
                'training_years': model_params.training_years
            })
            self.session.add(record)
            self.session.commit()
            if self.update_model_status(model_id = hash_id, status=1) == False:
                self.session.rollback()
                return False
        except Exception as ex:
            print(ex)
            self.session.rollback()
            return False
        finally:
            self.session.close()
            return True

    def update_model_status(self, model_id,  status=0):
        try:
            status_db = self.session.query(ModelStatus).filter(ModelStatus.model_id == model_id).first()
            if status_db != None:
                status_db.status = status
            else:
                record = ModelStatus(**{
                    'model_id': model_id,
                    'status': status
                })
                self.session.add(record)
            self.session.commit()
        except Exception as ex:
            self.session.rollback()
            print('[Update Model Status]\n', ex)
            return False
        finally:
            self.session.close()
            print('Update model %s as %s' %(model_id, status))
            return True

    def get_model_status(self, model_id):
        try:
            status_db = self.session.query(ModelStatus).filter(ModelStatus.model_id == model_id).first()
            if status_db != None:
                print('Status of %s: %s', status_db.model_id, status_db.status)
                return {
                'model_id':status_db.model_id,
                'status': status_db.status
                }
            else:
                self.update_model_status(model_id=model_id)
                return {
                'model_id':model_id,
                'status': 0
                }

        except Exception as ex:
            print(ex)
            return False
    
    def get_prediction_model(self, model_id):
        try:
            model = self.session.query(PredictionModel).filter(PredictionModel.model_id == model_id).first()
            if model == None:
                return None
            else:
                print('Model Queried Success:', model)
                return model
        except Exception as ex:
            print('Exception while geting predition model, mode_id: %s' %model_id, exec)
        finally:
            self.session.close()
            
    def get_prediction_models(self):
        try:
            query = self.session.query(PredictionModel)
            models=[]
            if models == None:
                return None
            else:
                models = pd.read_sql(query.statement, query.session.bind)
                print('Model List Queried Success:', models)
                return models
        except Exception as ex:
            print('Exception while geting predition model, mode_id')
        finally:
            self.session.close()
            return models

    def get_model_params(self, model_id):
        try:
            model = self.session.query(PredictionModel).filter(PredictionModel.model_id == model_id).first()
            if model == None:
                return None
            else:
                print('Model List Queried Success:', model)
        except Exception as ex:
            print('Exception while geting predition model, mode_id: %s' %model_id, exec)
        finally:
            self.session.close()
            return model