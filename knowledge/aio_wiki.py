# import aiohttp
# import asyncio
# import time
# from tqdm import tqdm

# # 请求限制：每秒最多 500 次
# MAX_REQUESTS_PER_SECOND = 2
URL = "https://en.wikipedia.org/w/api.php"

search_relative_topk = 50

# async def request_wiki_api(params, semaphore, rate_limiter):
#     """
#     异步请求 Wiki API，限制请求速率在 500 次/秒以内。
#     """
#     async with semaphore:  # 控制总的并发量
#         await rate_limiter()  # 限制每秒请求频率
#         async with aiohttp.ClientSession() as session:
#             async with session.get(URL, params=params) as response:
#                 response.raise_for_status()
#                 return await response.json()

# async def rate_limiter():
#     """每秒限制请求数"""
#     await asyncio.sleep(1.0 / MAX_REQUESTS_PER_SECOND)

def get_fuzzy_search_sent_params(sent, lang='en'):
    """构造模糊搜索句子的请求参数"""
    sent = sent[:300]  # api限制搜索长度
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": sent,
        "srlimit": search_relative_topk,
        "uselang": lang,
    }
    return params
    

# async def main():
#     semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_SECOND)  # 信号量绑定到当前事件循环

#     bs = 1000
#     input_file = 'data/wiki1m_for_simcse.txt'
#     with open(input_file, 'r', encoding='utf-8') as f:
#         sent_list = f.read().splitlines()

#     sent_list = sent_list[:100000]  # 只取前 1000 条数据

#     for i in tqdm(range(0, len(sent_list), bs), desc="Processing"):
#         sent_list_batch = sent_list[i:i+bs]
#         tasks = []
#         for sent in sent_list_batch:
#             params = get_fuzzy_search_sent_params(sent)
#             tasks.append(request_wiki_api(params, semaphore, rate_limiter))
#         print(f"{time.time()}:start")
#         results = await asyncio.gather(*tasks)
#         for index, result in enumerate(results):
#             # do something with result
#             print(f"{time.time()}:{index}")

# if __name__ == "__main__":
#     asyncio.run(main())

import httpx
import asyncio
import time
from tqdm import tqdm

class RateLimiter:
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_time = time.time()

    async def acquire(self):
        while self.tokens <= 0:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens += elapsed * self.rate
            self.last_time = now
            if self.tokens > self.rate:
                self.tokens = self.rate
            await asyncio.sleep(0.01)
        self.tokens -= 1

async def fetch_data(rate_limiter, client, url, params):
    await rate_limiter.acquire()
    response = await client.get(url, params=params)
    return response, time.time()

async def main():
    rate_limiter = RateLimiter(rate=10)  # 每秒限制请求数为 50

    bs = 1200
    input_file = 'data/wiki1m_for_simcse.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()

    sent_list = sent_list[:10000]  # 只取前 1000 条数据

    for i in tqdm(range(0, len(sent_list), bs), desc="Processing"):
        sent_list_batch = sent_list[i:i+bs]
        async with httpx.AsyncClient() as client:
            tasks = []
            for sent in sent_list_batch:
                params = get_fuzzy_search_sent_params(sent)
                # tasks.append(request_wiki_api(params, semaphore, rate_limiter))
                tasks.append(fetch_data(rate_limiter, client, URL, params))
            print(f"{time.time()}:start")
            results = await asyncio.gather(*tasks)
            for response, timestamp in results:
                print(f"{timestamp}: 1")

if __name__ == "__main__":
    asyncio.run(main())