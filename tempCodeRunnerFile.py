import os
from stocker_server.flask_api import app

#migration = Migration()
port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run()  