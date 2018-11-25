from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from stocker_app.stocker_server.endpoints import endpoints
from stocker_app.stocker_server.celery_init import make_celery
from stocker_app.stocker_server.intialize_endpoints import initialize_endpoints
from stocker_app.config.setting import configs
#code which helps initialize our server

app =  Flask(__name__)

app.config.update(
    CELERY_BROKER_URL='redis://:devpassword@redis:6379/0',
    CELERY_RESULT_BACKEND='redis://:devpassword@redis:6379/0'
)

postgre_connection_string=configs['postgre_connection_string']
DB_URL = os.getenv("DATABASE_URL",postgre_connection_string)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL

api = Api(app)

initialize_endpoints(api)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port) 