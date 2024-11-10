import requests
from tqdm import tqdm
import json
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# 连接 Redis 数据库
r = redis.Redis(host='localhost', port=6379, db=0, password='lyuredis579')

# 初始化计数器
empty_list_count = 0
total_count = 0
prefix = 'wikisearch:'

page_id_list_key = 'page_id_list'

if r.exists(page_id_list_key):
    page_id_list = json.loads(r.get(page_id_list_key))
else:
    page_id_list = []

    # 遍历符合条件的键并统计内容为空列表的键数量
    for key in tqdm(r.scan_iter(prefix + '*'), desc='Scan keys'):
        value = r.get(key)
        # 检查值是否为空列表
        if value is not None and value.decode() == '[]':
            empty_list_count += 1
        else:
            page_id = [item['page_id'] for item in json.loads(value)]
            page_id_list.extend(page_id)
        total_count += 1
    print(f"Keys with empty list content: {empty_list_count}")
    print(f"Total keys: {total_count}")

    # 去重
    page_id_list = list(set(page_id_list))
    print(f"Total page_id: {len(page_id_list)}")
    r.set(page_id_list_key, json.dumps(page_id_list))

r = redis.Redis(host='localhost', port=6379, db=1, password='lyuredis579')

proxy = {
    "https": "http://127.0.0.1:20171"
}

def get_detailed_page_info(page_id, language='en'):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "extracts|categories|info|images|pageprops|revisions",
        "explaintext": True,  # 返回纯文本格式
        "inprop": "url",      # 包含页面的URL信息
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, proxies=proxy)
        time.sleep(random.uniform(0.05, 0.2))
        data = response.json()
        
        page_data = data['query']['pages'][str(page_id)]
        
        # 将详细信息提取到字典中
        page_info = {
            "title": page_data.get("title"),
            "summary": page_data.get("extract"),  # 页面简介或全部内容
            "url": page_data.get("fullurl"),      # 页面URL
            "categories": [cat['title'] for cat in page_data.get("categories", [])],
            "images": [img['title'] for img in page_data.get("images", [])],  # 图片标题
            "wikidata_id": page_data.get("pageprops", {}).get("wikibase_item")
        }
        
        return page_info
    except Exception as e:
        print(f"Error querying page ID {page_id}: {e}")
        return None

prefix = 'wikipage:'

def get_page_info(page_id):
    key = prefix + str(page_id)
    if not r.exists(key):
        page_info = get_detailed_page_info(page_id)
        if page_info is not None:
            r.set(key, json.dumps(page_info, ensure_ascii=False))

# 使用线程池进行并发请求
max_workers = 30  # 设置并发线程数量
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(get_page_info, page_id) for page_id in page_id_list]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()  # 等待每个任务完成
