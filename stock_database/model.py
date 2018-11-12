from app import db
class Price_History(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False )
    ticker = db.Column(db.String(100))
    date = db.Column(db.Date)
    opn = db.Column(db.Float)
    hi = db.Column(db.Float)
    lo = db.Column(db.Float)
    close = db.Column(db.Float)
    vol = db.Column(db.Float)
