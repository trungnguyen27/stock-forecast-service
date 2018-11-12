import os
from flask import Flask
from flask_restful import Api
from stocker_server.flask_api import PriceData, MovingAverage, Prediction
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from global_configs import configs
from stocker_server.endpoints import endpoints

data_endpoint = endpoints['price_data']
ma_endpoint = endpoints['moving_average_data']
prediction_endpoint = endpoints['prediction']
#code which helps initialize our server
app =  Flask(__name__)
api = Api(app)

api.add_resource(PriceData, '%s/<string:ticker>' %data_endpoint)
api.add_resource(MovingAverage, '%s/<string:ticker>' %ma_endpoint)
api.add_resource(Prediction, '%s/<string:ticker>' %prediction_endpoint)

postgre_connection_string=configs['postgre_connection_string']
DB_URL = os.getenv("DATABASE_URL",postgre_connection_string)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
db = SQLAlchemy(app)
migrate = Migrate(app, db)

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run() 