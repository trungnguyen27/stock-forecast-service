import os
from flask_sqlalchemy import SQLAlchemy
from stocker_app.config import configs

class Database():
    def __init__(self, app):
        postgreurl = configs['postgre_connection_string']
        db_url = os.getenv("DATABASE_URL", postgreurl)
        app.config['SQLALCHEMY_DATABASE_URI'] = db_url
        try:
            self.db = SQLAlchemy(app)
        except Exception as ex:
            print(ex)
            self.initialized = False
        finally:
            self.initialized = True


    def get_session(self):
        if self.initialized == True:
            return self.db.session()
        print('Session is not initialized')
       
    def get_db_obj(self):
        if self.initialized == True:
            return self.db
    
    def get_model(self):
        return self.db.Model
    
    def create_tables(self):
        if self.initialized == True:
            self.db.create_all()