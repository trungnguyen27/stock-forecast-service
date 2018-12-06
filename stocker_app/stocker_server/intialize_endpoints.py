def initialize_endpoints(api):
    from stocker_app.stocker_server import endpoints, model_endpoints
    from stocker_app.stocker_server.flask_api import ModelBuild, PredictionAPI, ModelStatus, ModelList,  PriceData, MovingAverage, PredictionAPI, WelcomePage, DataMigration
    
    data_endpoint = endpoints['price_data']
    ma_endpoint = endpoints['moving_average_data']
    prediction_endpoint = endpoints['prediction']
    migration_endpoint= endpoints['migration']

    # Model
    model_build = model_endpoints['model_build']
    model_build_status = model_endpoints['model_build_status']
    model_list = model_endpoints['model_list']
    model_params = model_endpoints['model_params']

    api.add_resource(ModelStatus, '%s/<string:model_id>' %model_build_status)
    api.add_resource(ModelList, '%s' %model_list)
    api.add_resource(ModelBuild, '%s/<string:ticker>' %model_build)

    api.add_resource(PriceData, '%s/<string:ticker>' %data_endpoint)
    api.add_resource(MovingAverage, '%s/<string:ticker>' %ma_endpoint)
    api.add_resource(PredictionAPI, '%s' %prediction_endpoint)

    api.add_resource(WelcomePage, '/')
    api.add_resource(DataMigration, '%s/<int:start>' %migration_endpoint)