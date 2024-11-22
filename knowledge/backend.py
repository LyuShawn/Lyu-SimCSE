import redis

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # 将初始化参数作为实例的唯一标识
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[key] = instance
        return cls._instances[key]

class RedisClient(metaclass=SingletonMeta):

    password = 'lyuredis579'

    def __init__(self, host='59.77.134.205', port=6379, db=0):
        self._connection = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            password=self.password
        )
        
    def get_connection(self):
        return self._connection

    def get(self, key, *args, **kwargs):
        value = self._connection.get(key, *args, **kwargs)
        return value.decode() if value else None

    def mget(self, keys, *args, **kwargs):
        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list or tuple")
        values = self._connection.mget(keys, *args, **kwargs)
        return [value.decode() if value else None for value in values]