import sys
import os

# 将项目根目录加入 Python 搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from knowledge.backend import RedisClient
import json

search_relative_topk = 50
URL = "https://en.wikipedia.org/w/api.php"


def request_wiki_api(params):
    response = requests.get(URL, params=params)
    response.raise_for_status()
    return response.json()

def fuzzy_search_sent(sent, lang='en'):
    """模糊搜索句子"""

    sent = sent[:300]   # api限制搜索长度

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": sent,
        "srlimit": search_relative_topk,
        "uselang": lang,
    }

    response = request_wiki_api(params)

    if "query" not in response or "search" not in response["query"]:
        return []
    search_results = response["query"]["search"]
    return search_results

def page_info_retrieval(page_id, lang='en'):
    """检索页面详细信息"""
    #
    params = {
        "action": "query",
        "format": "json",
        "prop": "info|description|extracts|categories|langlinks|pageprops",
        "exintro": True,
        "explaintext": True,
        "pageids": page_id,
    }

    response = request_wiki_api(params)
    if "query" not in response or "pages" not in response["query"] or str(page_id) not in response["query"]["pages"]:
        return {}

    page_info = response["query"]["pages"][str(page_id)]
    return page_info

def main():
    max_workers = 8  # 设置并发线程数量
    input_file = 'data/wiki1m_for_simcse.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()

    redis_client = RedisClient()
    
    # sent_list = sent_list[:1000]

    # random.shuffle(sent_list)

    page_id_key = "wiki1m_page_id_list"
    page_id_list = redis_client.get(page_id_key)
    if not page_id_list:
        page_id_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fuzzy_search_sent, sent) for sent in sent_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Searching"):
                search_results = future.result()
                if search_results:
                    page_id_list+=[result['pageid'] for result in search_results]

        # 去重
        page_id_list = list(set(page_id_list))
    
        redis_client.set(page_id_key, page_id_list)
    else:
        page_id_list = json.loads(page_id_list)

    print(f"Total {len(page_id_list)} pages found!")

    # page_id_list = page_id_list[:100]
    random.shuffle(page_id_list)

    # 使用线程池进行并发请求
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(page_info_retrieval, page) for page in page_id_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    page_id = page_id_list[0]
    result = page_info_retrieval(page_id)
    print(result)
    print("Done!")

if __name__ == '__main__':
    main()
