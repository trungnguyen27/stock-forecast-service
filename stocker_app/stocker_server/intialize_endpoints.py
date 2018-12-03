def initialize_endpoints(api):
    from stocker_app.stocker_server.endpoints import endpoints
    from stocker_app.stocker_server.flask_api import PriceData, MovingAverage, Prediction, WelcomePage, DataMigration
    
    data_endpoint = endpoints['price_data']
    ma_endpoint = endpoints['moving_average_data']
    prediction_endpoint = endpoints['prediction']
    migration_endpoint= endpoints['migration']

    api.add_resource(PriceData, '%s/<string:ticker>' %data_endpoint)
    api.add_resource(MovingAverage, '%s/<string:ticker>' %ma_endpoint)
    api.add_resource(Prediction, '%s/<string:ticker>' %prediction_endpoint)

    api.add_resource(WelcomePage, '/')
    api.add_resource(DataMigration, '%s/<int:start>' %migration_endpoint)