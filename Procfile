web: gunicorn app:app --preload
worker: celery -A stocker_server.tasks.celery_app worker --loglevel=info