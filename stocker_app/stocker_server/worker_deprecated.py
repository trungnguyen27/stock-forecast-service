import os

import redis
from rq import Worker, Queue, Connection

from stocker_app.config import configs

listen = ['high', 'default', 'low']

redis_url = configs['redis']

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()