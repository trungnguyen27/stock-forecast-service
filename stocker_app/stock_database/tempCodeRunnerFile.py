from flask_sqlalchemy import SQLAlchemy, Model
from stocker_app.stock_database.database import Database

db = Database().get_db_obj()

class Price_History(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False )
    ticker = db.Column(db.String(100))
    date = db.Column(db.Date, )
    opn = db.Column(db.Float)
    hi = db.Column(db.Float)
    lo = db.Column(db.Float)
    close = db.Column(db.Float)
    vol = db.Column(db.Float)
    db.UniqueConstraint('ticker', 'date', 'ticker_date_uc')