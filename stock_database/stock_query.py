from global_configs import configs
import pandas as pd
from stock_database.model import Price_History
csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

class stockquery():
    def __init__(self):
        from app import db
        db_session = db.session()
        self.session = db_session
    def get_data(self, ticker = 'VIC'):
        print('Getting data of %s' %ticker)
        query = self.session.query(Price_History).filter(Price_History.ticker == ticker)
        df = pd.read_sql(query.statement, query.session.bind)
        return df