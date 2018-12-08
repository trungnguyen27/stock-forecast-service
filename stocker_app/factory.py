import os, logging
from flask import Flask
from stocker_app import celery
from stocker_app.utils.celery_util import init_celery
from stocker_app.config.setting import configs

def create_app(config=None, environment=None):
    app = Flask(__name__)
    app.config.update({'BROKER_URL':'redis://@redis:6379'})
    app.config.update(
        CELERY_BROKER_URL=configs['redis'],
        CELERY_RESULT_BACKEND=configs['redis']
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    postgre_connection_string=configs['postgre_connection_string']
    DB_URL = os.getenv("DATABASE_URL",postgre_connection_string)
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL

    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.DEBUG)
    app.logger.debug('this will show in the log')

    init_celery(app, celery)
    return app