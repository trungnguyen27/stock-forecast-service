import sys, os

parent_path = os.path.dirname(os.path.abspath('./'))

current_db_version = 1

configs = {
    "csv_path": "./stocker_app/stock_database/index_database",
    "sql_path": "./stocker_app/stock_database/price_history_db_v%d.db" %current_db_version,
    "postgre_path": "localhost/trungnuyen",
    "model_path": "./stocker_app/prediction_model",
    "metastock_name": "metastock_all_data_real",
    'postgre_connection_string':'postgres://stockeruser:stockeruser@stockerdb.csctqutzfgga.us-east-2.rds.amazonaws.com:5432/stockerdatabase' 
}
