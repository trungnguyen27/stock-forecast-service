from global_configs import configs
from time import time
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists
import pandas as pd
import csv
csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

Base = declarative_base()

class Price_History(Base):
    __tablename__ = 'Price_History'
    __table_args__ = {'sqlite_autoincrement':True}

    id = Column(Integer, primary_key=True, nullable=False )
    ticker = Column(String)
    date = Column(Date)
    opn = Column(Float)
    hi = Column(Float)
    lo = Column(Float)
    close = Column(Float)
    vol = Column(Float)

class Migration():
    def __init__(self):
        if not database_exists('sqlite:///%s' %sql_path):
            # data = pd.read_csv('%s/%s.csv' %(csv_path, metastock_name), parse_dates=[1], usecols = [0,1,2,3,4,5,6])
            # #data=data[data.columns[:7]]
            # print(data.describe())
            # data.columns= ["Ticker","Date","Open", "High", "Low", "Close", "Volume"]
            # data['Date'] = pd.to_datetime(data['Date']).apply(lambda x : x.date())
            # print('dataframe loaded')
            self.connect_database()
            self.load_csv()
            #self.data = data
            #self.init_database()
        else:
            self.connect_database()

    def load_csv(self):
        chunksize = 10000
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
        print('Connecting to database')
        engine = create_engine('sqlite:///%s' %sql_path)
        Base.metadata.create_all(engine)

        #create a session
        session = sessionmaker()
        session.configure(bind = engine)
        s = session()
        self.egnine = engine
        self.session = s

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
            s.commit()
            print("nana")
        except Exception as e:
            s.rollback()
            print(e)
        finally:
            s.close()
            print("time elapsed: %ss" %(str(time()-t)))

    def get_data(self, ticker = 'VIC'):
        query = self.session.query(Price_History).filter(Price_History.ticker == ticker)
        df = pd.read_sql(query.statement, query.session.bind)
        return df