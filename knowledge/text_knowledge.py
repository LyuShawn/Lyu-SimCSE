# 配置文件

db_config = {  
    'host':"59.77.134.205",
    'user':"root",
    'password':"lyumysql579",
    'database':"wikidata"
    }
es_config = {
    'hosts':"http://59.77.134.205:9200",
    'index' : 'wikidata-v1'
}
input_file = 'data/wiki1m_for_simcse.txt'
output_file = 'data/wiki1m_knowledge.txt'

# library
from elasticsearch import Elasticsearch
from tqdm import tqdm
import spacy
import mysql.connector
import json
import time

# 初始化模型和环境

# spacy
ner = spacy.load("en_core_web_sm")

# mysql
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# es
es = Elasticsearch(hosts=es_config['hosts'])

# dict
# GPE: 373869
# NORP: 152530
# ORG: 532072
# WORK_OF_ART: 80636
# FAC: 37106
# PERSON: 505931
# LOC: 56057
# PRODUCT: 22275
# LAW: 6742
# EVENT: 29668
# LANGUAGE: 8100
# sum: 1,763,006
    
# 事件:P31: Q1656682
# 地点:P31: Q2221906
# 语言:P31: Q34770
# 规章制度:P31: Q22097341
# 人物:P31: Q5
# 组织:P31: Q43229
# 产品:P31: Q2424752
# 艺术品:P31: Q838948

relation_dict = {
    'GPE' : {
        'P31' : ['Q2221906']
    },
    'NORP' : {
        'P31' : ['Q43229']
        },
    'ORG' : {
        'P31' : ['Q43229']
        },
    'WORK_OF_ART' : {
        'P31' : ['Q838948']
        },
    'FAC' : {
        'P31' : ['Q2221906']
        },
    'PERSON' : {
        'P31' : ['Q5']
        },
    'LOC' : {
        'P31' : ['Q2221906']
        },
    'PRODUCT' : {
        'P31' : ['Q2424752']
        },
    'LAW' : {
        'P31' : ['Q22097341']
        },
    'EVENT' : {
        'P31' : ['Q2221906']
        },
    'LANGUAGE' : {
        'P31' : ['Q34770']
        }
}

use_relation =['GPE','NORP','ORG','WORK_OF_ART','FAC','PERSON','LOC','PRODUCT','LAW','EVENT','LANGUAGE']

# es and mysql
def find_entity_description(entity_name,relation):
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "aliases.keyword": entity_name
                        }
                    },
                    {
                        "terms": {
                            "P31.keyword": relation
                        }
                    }
                ]
            }
        },
        "size": 1 
    }
    result = es.search(index=es_config['index'], body=query)
    return result

def find_entity_db(qid):
    sql = f'SELECT description FROM entity WHERE qid ={qid}'
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        return result[0][0]
    except Exception as e:
        print(f"error in {qid}:{e}")

es_hit = 0
db_hit = 0
with open(output_file, 'a', encoding='utf-8') as output_file, open(input_file, 'r', encoding='utf-8') as file:
    for index, line in tqdm(enumerate(file),total=1000000):
        try:
            sentenct_info = {}
            sentenct_info['sent_index'] = index
            sentenct_info['sent'] = line.strip()
            words = line.strip()
            doc = ner(line)
            ent_info_list = []
            for ent in doc.ents:
                if ent.label_ not in use_relation:
                    continue
                entity_info = {}
                entity_info['entity'] = ent.text
                entity_info['entity_type'] = ent.label_
                entity_info['start_char'] = ent.start_char
                entity_info['end_char'] = ent.end_char
            
                # es
                es_result = find_entity_description(entity_info['entity'],relation_dict[entity_info['entity_type']]['P31'])
                if (len(es_result['hits']['hits'])) > 0:
                    result  = es_result['hits']['hits'][0]['_source']
                    entity_info['qid'] = result['qid']
                    entity_info['aliases'] = result['aliases']
                    entity_info['P31'] = result['P31']
                    es_hit+=1
                
                if 'qid' in entity_info:
                    entity_info['description'] = find_entity_db(entity_info['qid'])
                    db_hit +=1
                ent_info_list.append(entity_info)
            if len(ent_info_list) > 0:
                sentenct_info['entities'] = ent_info_list
            json.dump(sentenct_info,output_file)
            output_file.write("\n")
        except Exception as e:
            print(f"error in {index}:{e}")
            continue
            
