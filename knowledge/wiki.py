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
from knowledge.backend import MySQLClient

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
        "prop": "categories",  # 获取页面的分类
        "cllimit": "max",      # 限制返回的分类数量，这里设定为最大
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
    input_file = 'data/wiki1m_for_simcse.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()
    print(f"Total sentences: {len(sent_list)}")

    sent_list = sent_list[0:1000]
    for sent in tqdm(sent_list):
        search_results = fuzzy_search_sent(sent)



if __name__ == '__main__':
    main()
