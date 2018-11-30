from stocker_app.stock_database.database import Database
from stocker_app.application import app

database = Database(app)
db = database.get_db_obj()

print('RUNN')

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
    
class App_Setting(db.Model):
    __table_name__ = 'app_setting'
    key = db.Column(db.String(100), primary_key =True, nullable = False)
    value = db.Column(db.Integer, nullable = False)