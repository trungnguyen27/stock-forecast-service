from stocker_app.config.setting import configs
import pandas as pd
from stocker_app.stock_database.schemas import database, Price_History
csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

class stockquery():
    def __init__(self):
        db_session = database.get_session()
        self.session = db_session
    def get_data(self, ticker = 'VIC'):
        print('Getting data of %s' %ticker)
        query = self.session.query(Price_History).filter(Price_History.ticker == ticker)
        df = pd.read_sql(query.statement, query.session.bind)
        return df