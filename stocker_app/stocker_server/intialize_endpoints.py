def initialize_endpoints(api):
    from stocker_app.stocker_server.endpoints import endpoints
    from stocker_app.stocker_server.flask_api import PriceData, MovingAverage, Prediction
    
    data_endpoint = endpoints['price_data']
    ma_endpoint = endpoints['moving_average_data']
    prediction_endpoint = endpoints['prediction']
    
    api.add_resource(PriceData, '%s/<string:ticker>' %data_endpoint)
    api.add_resource(MovingAverage, '%s/<string:ticker>' %ma_endpoint)
    api.add_resource(Prediction, '%s/<string:ticker>' %prediction_endpoint)