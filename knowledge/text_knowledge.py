
# 每条句子搜索wiki
import redis
import base64
from tqdm import tqdm
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

r = redis.Redis(host='localhost', port=6379, db=0, password='lyuredis579')

prefix = 'wikisearch:'

api_url = 'https://en.wikipedia.org/w/api.php'

headers = {
    "User-Agent": "Wiki Study/1.0 (905899183@qq.com)"
}

# 从环境变量中获取代理信息
import os
proxy = {
    "http": os.getenv("HTTP_PROXY", ""),
    "https": os.getenv("HTTPS_PROXY", "")
}

def search_wiki(text):
    # 搜出10个结果
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{text}"',  # 精确匹配内容
        "srwhat": "text",         # 指定在内容中搜索
        "format": "json",
    }

    response = requests.get(api_url, params=params, headers=headers, proxies=proxy)
    # time.sleep(0.1)
    search_results = response.json().get("query", {}).get("search", [])
    
    result = [{"page_id": result["pageid"], "title": result["title"]} for result in search_results]
    return result

def text_search(text):
    key = prefix + text_encode(text)

    cached_data= r.get(key)
    if cached_data:
        return json.loads(cached_data)
    else:
        try:
            result = search_wiki(text)
            result = json.dumps(result, ensure_ascii=False)
            r.set(key, result)
        except Exception as e:
            print(f"Error querying {text}: {e}")
            return None

def text_encode(text):
    # base64 编码
    return base64.b64encode(text.encode()).decode()
def text_decode(text):
    # base64 解码
    return base64.b64decode(text.encode()).decode()

dataset_path = '../data/wiki1m_for_simcse.txt'
with open(dataset_path, 'r', encoding='utf-8') as file:
    sent_list = file.read().splitlines()

# 使用线程池进行并发请求
max_workers = 10  # 设置并发线程数量
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(text_search, sent) for sent in sent_list]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()  # 等待每个任务完成