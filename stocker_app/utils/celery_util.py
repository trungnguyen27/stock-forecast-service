from stocker_app.stocker_server import celeryconfig
def init_celery(app, celery):
    celery.conf.update(app.config)
    #celery.config_from_object(celeryconfig)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask