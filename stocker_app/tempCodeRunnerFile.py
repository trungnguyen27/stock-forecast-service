from flask import Flask
from stocker_app import celery
from stocker_app.utils.celery_util import init_celery
def create_app(config=None, environment=None):
    app = Flask(__name__)
    app.config.update({'CELERY_BROKER_URL':'redis://localhost:6379'})
    init_celery(app, celery)
    return app