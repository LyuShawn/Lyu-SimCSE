import sys
import os

# 将项目根目录加入 Python 搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import random
import requests
from utils.cache_util import two_level_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

search_relative_topk = 50
URL = "https://en.wikipedia.org/w/api.php"

@two_level_cache
def request_wiki_api(params):
    response = requests.get(URL, params=params)
    response.raise_for_status()
    return response.json()

def fuzzy_search_sent(sent, lang='en'):
    """模糊搜索句子"""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": sent,
        "srlimit": search_relative_topk,
        "uselang": lang,
    }

    response = request_wiki_api(params)
    try:
        search_results = response["query"]["search"]
        return search_results
    except Exception:
        pass
        return []


def sent_info_retrieval(sent, lang='en', topk=5):
    """句子信息搜索"""
    # 1、模糊搜索相关的句子

def main():
    input_file = 'data/wiki1m_for_simcse.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()
    random.shuffle(sent_list)

    # 使用线程池进行并发请求
    max_workers = 16  # 设置并发线程数量
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fuzzy_search_sent, sent, max_workers) for sent in sent_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

if __name__ == '__main__':
    main()

