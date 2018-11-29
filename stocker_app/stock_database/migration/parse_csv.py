import pandas as pd
import csv, os
from time import time
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists
from stocker_app.config.setting import configs


csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

class Migration():
    def __init__(self):
        print('Looking for %s/%s.csv' %(csv_path, metastock_name))
        if os.path.isfile('%s/%s.csv' %(csv_path, metastock_name)) == True:
            print('Found')
            self.file_found = True
        else:
            print('Designated CSV file not found, migration may not be continued')
            self.file_found = False
        try:
            from stocker_app.stock_database.schemas import database
            self.session = database.get_session()
            database.create_tables()
        except Exception as ex:
            print(ex)
        finally:
            print('Migration object initialized')

    def migrate(self):
        if not self.file_found:
            print('Cannot load CSV, File Not Found')
            return
        # self.connect_database()
        self.load_csv()
            
    def load_csv(self):
        chunksize = 50
        index = 0
        try:
            for chunk in pd.read_csv('%s/%s.csv' %(csv_path, metastock_name), chunksize= chunksize, parse_dates=[1], usecols = [0,1,2,3,4,5,6]):
                chunk.columns= ["Ticker","Date","Open", "High", "Low", "Close", "Volume"]
                chunk['Date'] = pd.to_datetime(chunk['Date']).apply(lambda x : x.date())
                self.save_to_database(chunk)
                index = index + chunksize
                print('Current Index: %d' %(index))
        except Exception as ex:
            print('[Exception]- load_csv:', ex)
        finally:
            print('done reading csv')

    # def connect_database(self):
    #     try:
    #         self.database.create_tables()
    #     except Exception as ex:
    #         print(ex)
    #     finally:
    #         print('Connected to PostgreSQL')

    def save_to_database(self, data):
        t = time()
        s = self.session
        from stocker_app.stock_database.schemas import Price_History
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
            print('Committed', data.head())
        except Exception as e:
            s.rollback()
            print('[Exception|save_to_database]', e)
        finally:
            s.close()
            print("Time elapsed: %ss" %(str(time()-t)))

    def get_data(self, ticker = 'VIC'):
        print('Getting data of %s' %ticker)
        from stocker_app.stock_database.schemas import Price_History
        query = self.session.query(Price_History).filter(Price_History.ticker == ticker)
        df = pd.read_sql(query.statement, query.session.bind)
        return df