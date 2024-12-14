import diskcache as dc

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