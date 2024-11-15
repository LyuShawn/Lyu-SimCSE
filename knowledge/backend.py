import redis

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class RedisClient(metaclass=SingletonMeta):

    host = '59.77.134.205'
    port = 6379
    db = 0
    password = 'lyuredis579'

    def __init__(self):
        self._connection = redis.StrictRedis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password
        )
        
    def get_connection(self):
        return self._connection

    def get(self, key, *args, **kwargs):
        value = self._connection.get(key, *args, **kwargs)
        return value.decode() if value else None