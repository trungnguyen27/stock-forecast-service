from stocker_server.endpoints import endpoints
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from global_configs import configs
from stocker_server.celery_init import make_celery
import os
from stocker_server.intialize_endpoints import initialize_endpoints
#code which helps initialize our server
app =  Flask(__name__)

app.config.update(
    CELERY_BROKER_URL=os.getenv('REDIS_URL','redis://localhost:6379'),
    CELERY_RESULT_BACKEND=os.getenv('REDIS_URL','redis://localhost:6379')
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
    app.run() 