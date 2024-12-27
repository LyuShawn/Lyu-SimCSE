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
import json

search_relative_topk = 50
URL = "https://en.wikipedia.org/w/api.php"

@two_level_cache
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
    try:
        search_results = response["query"]["search"]
        return search_results
    except Exception:
        return None

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
    try:
        page_info = response["query"]["pages"][str(page_id)]
        return json.dumps(page_info, indent=4)
    except Exception:
        pass
        return {}

def main():
    max_workers = 8  # 设置并发线程数量
    input_file = 'data/wiki1m_for_simcse.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()

    random.shuffle(sent_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fuzzy_search_sent, sent) for sent in sent_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    page_id_list = []
    for sent in tqdm(sent_list):
        search_results = fuzzy_search_sent(sent)
        if search_results:
            page_id_list+=[result['pageid'] for result in search_results]    
    # 去重
    page_id_list = list(set(page_id_list))

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
