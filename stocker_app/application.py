from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
import os, logging
from stocker_app.stocker_server.endpoints import endpoints
from stocker_app.stocker_server.celery_init import make_celery
from stocker_app.stocker_server.intialize_endpoints import initialize_endpoints
from stocker_app.config.setting import configs


#code which helps initialize our server

app =  Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)
app.logger.debug('this will show in the log')

app.config.update(
    CELERY_BROKER_URL=configs['redis'],
    CELERY_RESULT_BACKEND=configs['redis']
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

postgre_connection_string=configs['postgre_connection_string']
DB_URL = os.getenv("DATABASE_URL",postgre_connection_string)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL

api = Api(app)

initialize_endpoints(api)

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
