from stocker_app.stock_database.migration.parse_csv import Migration
m = Migration()
df = m.get_data(ticker='000001.SS')
print(df)