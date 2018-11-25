from config.setting import configs
from time import time
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists
import pandas as pd
import csv
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from stocker_app.stocker_server.flask_api import app
DB_URL = os.getenv("DATABASE_URL","postgres://pkfjujpoljriaq:7527e2b110024e147f74c8d05d61e6a793a504814adef500bc976c101f1ada94@ec2-54-225-115-234.compute-1.amazonaws.com:5432/d3umils42h7mt5")
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL

db = SQLAlchemy(app)
migrate = Migrate(app, db)

db_session = db.session()


csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

class Price_History(db.Model):
    # __tablename__ = 'Price_History'
    # __table_args__ = {'sqlite_autoincrement':True}
    id = db.Column(db.Integer, primary_key=True, nullable=False )
    ticker = db.Column(db.String(100))
    date = db.Column(db.Date)
    opn = db.Column(db.Float)
    hi = db.Column(db.Float)
    lo = db.Column(db.Float)
    close = db.Column(db.Float)
    vol = db.Column(db.Float)

class Migration():
    def __init__(self):
        
        self.connect_database()
        # try:
        #     db.create_all()
        # except Exception as ex:
        #     print(ex)
        # finally:
        #     self.load_csv()
        # if not database_exists(postgre_url):
        #     print('NOT EXISTS')
        #     # data = pd.read_csv('%s/%s.csv' %(csv_path, metastock_name), parse_dates=[1], usecols = [0,1,2,3,4,5,6])
        #     # #data=data[data.columns[:7]]
        #     # print(data.describe())
        #     # data.columns= ["Ticker","Date","Open", "High", "Low", "Close", "Volume"]
        #     # data['Date'] = pd.to_datetime(data['Date']).apply(lambda x : x.date())
        #     # print('dataframe loaded')
        #     self.connect_database()
        #     self.load_csv()
        #     #self.data = data
        #     #self.init_database()
        # else:
            

    def load_csv(self):
        chunksize = 50
        try:
            for chunk in pd.read_csv('%s/%s.csv' %(csv_path, metastock_name), chunksize= chunksize, parse_dates=[1], usecols = [0,1,2,3,4,5,6]):
                chunk.columns= ["Ticker","Date","Open", "High", "Low", "Close", "Volume"]
                chunk['Date'] = pd.to_datetime(chunk['Date']).apply(lambda x : x.date())
                self.save_to_database(chunk)
        except Exception as ex:
            print(ex)
        finally:
            print('done reading csv')

    def connect_database(self):
        
         # create the database
        # print('Connecting to database')
        # engine = create_engine(postgre_url)
        # Base.metadata.create_all(engine)

        # #create a session
        # session = sessionmaker()
        # session.configure(bind = engine)
        # # s = session()
        # self.egnine = engine
        self.session = db_session

    def save_to_database(self, data):
        t = time()
        s = self.session
        try:
            for (index, i) in data.iterrows():
                record = Price_History(**{
                    'date':i['Date'],
                    'ticker': i['Ticker'],
                    'opn':i['Open'],
                    'hi': i['High'],
                    'lo':i['Low'],
                    'close':i['Close'],
                    'vol':i['Volume']
                })
                #add all the records
                s.add(record) 
                if index >= 900:
                    break
                print(index)
            s.commit()
            print("nana")
        except Exception as e:
            s.rollback()
            print(e)
        finally:
            s.close()
            print("time elapsed: %ss" %(str(time()-t)))

    def get_data(self, ticker = 'VIC'):
        print('Getting data of %s' %ticker)
        query = self.session.query(Price_History).filter(Price_History.ticker == ticker)
        df = pd.read_sql(query.statement, query.session.bind)
        return df