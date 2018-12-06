model = '/model'
prediction = '/prediction'

endpoints = {
    'price_data': '/data/price',
    'moving_average_data': '/data/ma',
    'prediction': '%s/get' %prediction,
    'migration':'/migrate',
}

model_endpoints = {
    'model_build': '%s/build' %model,
    'model_build_status': '%s/status' %model,
    'model_list': '%s/all' %model,
    'model_params': '%s/params' %model,
}
