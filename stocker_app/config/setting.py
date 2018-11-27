import sys, os

parent_path = os.path.dirname(os.path.abspath('./'))

current_db_version = 1

configs = {
    "csv_path": "./stocker_app/stock_database/index_database",
    "sql_path": "./stocker_app/stock_database/price_history_db_v%d.db" %current_db_version,
    "postgre_path": "localhost/trungnuyen",
    "model_path": "./stocker_app/prediction_model",
    "metastock_name": "metastock_all_data_real",
    'postgre_connection_string':'postgres://pkfjujpoljriaq:7527e2b110024e147f74c8d05d61e6a793a504814adef500bc976c101f1ada94@ec2-54-225-115-234.compute-1.amazonaws.com:5432/d3umils42h7mt5' 
}
