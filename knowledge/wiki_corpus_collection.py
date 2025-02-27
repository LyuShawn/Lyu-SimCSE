import requests
import random
import mwparserfromhell
import re
from backend import RedisClient, MySQLClient
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import json
import argparse
import time

WIKI_API = "https://en.wikipedia.org/w/api.php"
LIMIT = 500
USER_AGENT = "SentenceFromWiki (lyushawn@foxmail.com)"
header = { 'User-Agent': USER_AGENT }
nltk.download('punkt')

# r = RedisClient(db=3)   # 使用第3个数据库存储页面信息
MySQL = MySQLClient()

# 获取关键词对应的页面URL
params_base = {
    'action': 'query',
    'list': 'search',
    'srsearch': "biomedical",
    'format': 'json',
    'srlimit': LIMIT,  # 设置每次请求获取的最大页面数
    'sroffset': 0,
}

def search_page(keyword, offset=0):
    """根据keyword搜索，返回page_id_list"""
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': keyword,
        'format': 'json',
        'srlimit': LIMIT,  # 设置每次请求获取的最大页面数
        'sroffset': offset,
    }
    try:
        response = requests.get(WIKI_API, params=params, headers=header)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch api with error code: {response.status_code}")
        data = response.json()
        total_hits = data['query']['searchinfo']['totalhits']
        next_offset = data['continue']['sroffset'] if 'continue' in data else None
        page_id_list = [page['pageid'] for page in data['query']['search']]
        return total_hits, page_id_list,next_offset
    except Exception as e:
        print(e)
        return 0, [], 1

def parse_wiki_text(wiki_text):
    """处理wikitext形式的文本"""
    # 使用 mwparserfromhell 解析 Wiki 格式文本
    parsed = mwparserfromhell.parse(wiki_text)

    # 提取 Wiki 格式文本中的纯文本
    plain_text = parsed.strip_code()

    # 去除文本中的换行符 (\n) 和制表符 (\t)
    plain_text = re.sub(r'[\n\t]+', ' ', plain_text)  # 将换行符和制表符替换为单个空格

    # 去除以 Category: 开头的内容
    plain_text = re.sub(r'\[\[Category:[^]]*\]\]', '', plain_text)
    pain_text= plain_text.split("Category:")[0]

    # 去除多余的空格
    plain_text = re.sub(r'\s+', ' ', pain_text).strip()  # 将多个空白字符合并为一个空格
    return plain_text

def get_page_info(page_id_list):
    """获取页面详细信息"""
    if len(page_id_list) > 50:
        raise Exception("The number of page ids exceeds the limit of 50.")

    # 获取页面详细信息
    params = {
        "action": "query",  # 查询操作
        "prop": "revisions|categories",  # 获取页面内容和类别
        "pageids": "|".join(map(str, page_id_list)),  # 以|分隔的页面ID列表
        "rvslots": "main",  # 获取主内容
        "format": "json",  # 返回JSON格式
        "rvprop": "content",  # 获取内容（正文部分）
        "utf8": 1  # 使用UTF-8编码
    }

    try:
        start_time = time.time()

        response = requests.get(WIKI_API, params=params, headers=header)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch api with error code: {response.status_code}")
        data = response.json()
        if 'query' not in data:
            raise Exception(f"No query in data with : {data}")
        page_info_list_raw = data['query']['pages']
        page_info_dict = {}
        for page_id, page_info_raw in page_info_list_raw.items():
            content = page_info_raw['revisions'][0]['slots']['main']['*']
            page_info_dict[page_id] = parse_wiki_text(content)

        time_used = time.time() - start_time
        return page_info_dict, time_used
    except Exception as e:
        print(e)
        return {}, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="Medicine")
    parser.add_argument("--domain", type=str, default="Medicine")

    args = parser.parse_args()

    keyword = args.keyword
    domain = args.domain

    offset = 0
    pbar = tqdm(total=100000,desc="collecting sentences")
    while True:
        total_hits, page_id_list, next_offset = search_page(keyword, offset)
        pbar.total = total_hits
        if not next_offset:
            break
        offset = next_offset

        exist_keys, non_exist_keys = MySQL.batch_page_content_id_exist(page_id_list)

        # 不存在的keys按50一组划分
        for i in range(0, len(non_exist_keys), 50):
            page_info_dict, used_time = get_page_info(non_exist_keys[i:i+50])
            MySQL.batch_set_wiki_page_content(page_info_dict, keyword, domain)
            pbar.set_description(f"total:{total_hits} collecting from wiki: {used_time:.2f}s")
        pbar.update(len(page_id_list))


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--keyword", type=str, default="biomedical")

#     args = parser.parse_args()

#     keyword = args.keyword

#     sent_collect_num = 1_000_000
#     # 句子获取倍数
#     sent_multiple = 3
#     max_sent_num = sent_collect_num * sent_multiple
#     output_file = f"data/{keyword}_wiki1m.txt"

#     sent_list = []
#     offset = 0
#     pbar = tqdm(total=max_sent_num,desc="collecting sentences")
#     while len(sent_list) < max_sent_num:
#         total_hits, page_id_list = search_page(keyword, offset)
#         offset += LIMIT
#         if offset > total_hits:
#             break

#         exist_keys, non_exist_keys = MySQL.batch_page_content_id_exist(page_id_list)

#         sent_list_tmp = []

#         # 存在的keys
#         page_info_dict = MySQL.batch_get_wiki_page_content(exist_keys)
#         for page_id, page_info in page_info_dict.items():
#             sent_list_tmp += sent_tokenize(page_info)
#             pbar.set_description(f"total:{total_hits} collecting from mysql")

#         # 不存在的keys按50一组划分
#         for i in range(0, len(non_exist_keys), 50):
#             page_info_dict, used_time = get_page_info(non_exist_keys[i:i+50])
#             MySQL.batch_set_wiki_page_content(page_info_dict, keyword)
#             for page_id, page_content in page_info_dict.items():
#                 sent_list_tmp += sent_tokenize(page_content)
#             pbar.set_description(f"total:{total_hits} collecting from wiki: {used_time:.2f}s")

#         # 这里过滤句子
#         # TODO
#         sent_list += sent_list_tmp
#         pbar.update(len(sent_list_tmp))

#     # 采样
#     if len(sent_list) > sent_collect_num:
#         sent_list = random.sample(sent_list, sent_collect_num)

#     with open(output_file, 'w') as f:
#         for sent in sent_list:
#             f.write(sent + "\n")


if __name__ == '__main__':
    main()