import diskcache as dc
from knowledge.backend import RedisClient
import json

# 创建磁盘缓存
cache = dc.Cache('cache')

def disk_cache(func):
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}:{args}:{kwargs}"
        if cache_key in cache:
            # print("Cache hit!")
            return cache[cache_key]
        # print("Cache miss!")
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result
    return wrapper

def two_level_cache(func):
    """函数二级缓存装饰器"""
    def wrapper(*args, **kwargs):
        # 序列化参数
        cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        if cache_key in cache:
            # 文件缓存命中
            return cache[cache_key]
        try:
            # 文件缓存未命中，尝试Redis缓存
            redis_client = RedisClient()
            value = redis_client.get(cache_key)
            if value:
                result = json.loads(value)
                # 二级缓存更新一级缓存
                cache.set(cache_key, result)
                return result
        except Exception as e:
            # 异常说明没有Redis服务
            pass

        # 二级缓存未命中，调用函数
        result = func(*args, **kwargs)
        # 更新二级缓存
        cache.set(cache_key, result)
        try:
            redis_client.set(cache_key, result)
        except Exception as e:
            # 异常说明没有Redis服务
            pass

        return result
    return wrapper