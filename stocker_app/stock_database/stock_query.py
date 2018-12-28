from stocker_app.config import configs
import pandas as pd
from stocker_app.stock_database.schemas import database, Price_History
csv_path = configs['csv_path']
sql_path = configs['sql_path']
metastock_name = configs['metastock_name']

class stockquery():
    def __init__(self):
        db_session = database.get_session()
        self.session = db_session
   