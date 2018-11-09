from stocker_server import app
from stock_database.parse_csv import Migration

migration = Migration()
app.run(debug=False)