import sys, os

parent_path = os.path.dirname(os.path.abspath('./'))

current_db_version = 1


configs = {
    "csv_path": "./stock_database/index_database",
    "sql_path": "./stock_database/price_history_db_v%d.db" %current_db_version,
    "postgre_path": "localhost/trungnuyen",
    "model_path": "./prediction_model",
    "metastock_name": "metastock_all_pvc"
}
