from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
import os, logging
from stocker_app.stocker_server.endpoints import endpoints
from stocker_app.stocker_server.intialize_endpoints import initialize_endpoints
from stocker_app.config.setting import configs
from stocker_app.factory import create_app
from stocker_app import celery
from stocker_app.utils.celery_util import init_celery

app = create_app()
init_celery(app, celery)
api = Api(app)

initialize_endpoints(api)

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
