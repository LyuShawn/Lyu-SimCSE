# 计算title和abstract的长度
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
from knowledge.backend import MySQLClient
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer
import json

data_dir = 'draw-data/'
if os.path.exists(data_dir) == False:
    os.makedirs(data_dir)

nlp = spacy.load('en_core_web_sm')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

print('开始加载数据')
# with open(data_dir + 'c4-title.txt', 'r', encoding='utf-8') as f:
#     title_list = f.read().splitlines()
with open(data_dir + 'f-c4-abstract.json', 'r', encoding='utf-8') as f:
    abstract_list = json.load(f)
print('加载数据完成')

# 计算title
batch_size = 500
n_p = 10
# docs = list(tqdm(nlp.pipe(title_list, batch_size=batch_size, disable=["ner", "parser", "textcat"], n_process=n_p), total=len(title_list),desc='计算title句子长度'))
# title_l_list = [len(doc) for doc in docs]

# title_sent_l_arr = np.array(title_l_list)
# np.save(data_dir + 'c4-title句子长度数组.npy', title_sent_l_arr)

# title_token_l_list = []

# for sent in tqdm(title_list, desc='计算title句子token长度'):
#     title_token_l_list.append(len(tokenizer.tokenize(sent)))

# title_token_l_arr = np.array(title_token_l_list)
# np.save(data_dir + 'c4-title句子token长度数组.npy', title_token_l_arr)

# 计算abstract

# 每5000个batch一次
abstract_l_list = []
for i in tqdm (range(0, len(abstract_list), 5000), desc='计算abstract句子长度'):
    docs = list(tqdm(nlp.pipe(abstract_list[i:i+5000], batch_size=batch_size, disable=["ner", "parser", "textcat"], n_process=n_p), total=len(abstract_list[i:i+5000]),desc='计算batch内abstract句子长度'))
    abstract_l_list += [len(doc) for doc in docs]

# docs = list(tqdm(nlp.pipe(abstract_list, batch_size=batch_size, disable=["ner", "parser", "textcat"], n_process=n_p), total=len(abstract_list),desc='计算abstract句子长度'))
# abstract_l_list = [len(doc) for doc in docs]

abstract_sent_l_arr = np.array(abstract_l_list)
np.save(data_dir + 'c4-abstract句子长度数组.npy', abstract_sent_l_arr)

abstract_token_l_list = []

for sent in tqdm(abstract_list, desc='计算title句子token长度'):
    abstract_token_l_list.append(len(tokenizer.tokenize(sent)))

abstract_token_l_arr = np.array(abstract_token_l_list)
np.save(data_dir + 'c4-abstract句子token长度数组.npy', abstract_token_l_arr)