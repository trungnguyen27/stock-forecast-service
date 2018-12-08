from stocker_app.stock_database import Database
from stocker_app.application import app
from stocker_app.factory import create_app

database = Database(create_app())
db = database.get_db_obj()

print('INTIALIZING SCHEMAS')

class Price_History(db.Model):
    __table_name__ = 'price_history'
    __table_args__ = (
       db.UniqueConstraint('ticker', 'date', name= '_ticker_date_uc'),
    )
    id = db.Column(db.Integer, primary_key=True, nullable=False )
    ticker = db.Column(db.String(100))
    date = db.Column(db.Date )
    opn = db.Column(db.Float)
    hi = db.Column(db.Float)
    lo = db.Column(db.Float)
    close = db.Column(db.Float)
    vol = db.Column(db.Float)

class PredictionModel(db.Model):
    __table_name__='prediction_model'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(64), db.ForeignKey('model_status.model_id'), nullable=False, unique=True )
    model_pkl = db.Column(db.PickleType)
    ticker = db.Column(db.String(4), nullable=False)
    prediction_start = db.Column(db.Date, nullable=False)
    changepoint_prior_scale = db.Column(db.Float)
    lag = db.Column(db.Integer)
    daily_seasonality = db.Column(db.Boolean)
    weekly_seasonality = db.Column(db.Boolean)
    monthly_seasonality = db.Column(db.Boolean)
    yearly_seasonality = db.Column(db.Boolean)
    quarterly_seasonality = db.Column(db.Boolean)
    training_years = db.Column(db.Integer)

class ModelStatus(db.Model):
    __table_name__='model_status'
    id = db.Column(db.Integer, primary_key = True, nullable=False)
    model_id = db.Column(db.String(100),
                    nullable=False,
                    unique=True)
    status = db.Column(db.Integer, nullable = False)
    prediction_model = db.relationship('PredictionModel', backref = 'modelstatus', lazy=False, uselist=False)
    
# class User(db.Model):
#     __table_name__= 'users'
#     id = db.Column(db.Integer, primary_key = True, nullable=False)
#     username= db.Column(db.String(20), nullable=False)
#     password = db.Column(db.String(64), nullable=True)

class App_Setting(db.Model):
    __table_name__ = 'app_setting'
    key = db.Column(db.String(100), primary_key =True, nullable = False)
    value = db.Column(db.Integer, nullable = False)

