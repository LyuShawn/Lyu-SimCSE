# 
import requests
from tqdm import tqdm
from knowledge.backend import MySQLClient

MySQL = MySQLClient()

WIKI_API = "https://en.wikipedia.org/w/api.php"

total = MySQL.get_sent_page_in_page_id_num()
pbar = tqdm(total=total)
offset = 0
limit=  1000
lang = 'en'

def get_page_info(page_id_list):
    
    page_id_list = [str(page_id) for page_id in page_id_list]

    params = {
            "action": "query",
            "pageids": "|".join(page_id_list),         # 指定页面ID
            "prop": "info|categories|extracts",  # 同时获取基础信息和分类数据
            "format": "json",           # 返回JSON格式
            "inprop": "url",            # 包含页面URL信息[[23]]
            "cllimit": "max",           # 获取最多500个分类（API限制）[[4,23]]
            "clshow": "!hidden",        # 排除隐藏分类[[23]]
            "exintro": True,         # 仅提取页面的简介
            "explaintext": True,     # 返回纯文本，不包含HTML
        }

    response = requests.get(WIKI_API, params=params, timeout=10)
    data = response.json()
    page_info_list = []
    if 'query' in data and 'pages' in data['query']:
        pages = data['query']['pages']
        for page in pages.values():
            page_info = {}
            page_info['page_id'] = page.get("pageid")
            page_info['title'] = page.get('title')
            page_info['full_url'] = page.get('fullurl')

            category_list = page.get('categories', [])
            page_info['categories'] = '|'.join([category['title'] for category in category_list])
            page_info['abstract'] = page.get('extract', '')

            page_info_list.append(page_info)
    return page_info_list

while True:
    page_list = MySQL.batch_get_sent_page_in_page_id_random(limit=limit)

    page_info_size = MySQL.get_page_info_size()

    if not page_list or page_info_size >= total:
        print('done')
        break
    offset += limit
    pbar.update(len(page_list))

    exist_keys, non_exist_keys = MySQL.page_info_exist(page_list,lang)

    # 50个一组
    bs = 20
    for i in range(0, len(non_exist_keys), bs):
        keys = non_exist_keys[i:i+bs]
        try:
            page_info_list = get_page_info(keys)
            MySQL.batch_insert_page_info(page_info_list, lang)
        except Exception as e:  # 有可能是网络问题
            print(e)
            continue