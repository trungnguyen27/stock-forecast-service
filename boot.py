from stocker_server import app
from stock_database.parse_csv import Migration
import os

migration = Migration()
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)