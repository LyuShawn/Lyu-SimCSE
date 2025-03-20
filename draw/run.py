# 计算title和abstract的长度
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
from knowledge.backend import MySQLClient
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer

data_dir = 'draw-data/'

nlp = spacy.load('en_core_web_sm')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

MySQL = MySQLClient()

title_list = []
abstrct_list = []
offset = 0
limit = 1000
print('开始加载数据')
while True:
    sent_list = MySQL.batch_get_page_info_title_abstract(offset,limit)
    offset+=limit
    # if not sent_list:
    #     break
    for sent in sent_list:
        title_list.append(sent[1])
        abstrct_list.append(sent[2])

    # 每10万打印一次
    if offset % 100000 == 0:
        print(f'已加载{offset}条数据')
        break

# 计算title
batch_size = 5000
docs = list(tqdm(nlp.pipe(title_list, batch_size=batch_size, disable=["ner", "parser", "textcat"], n_process=10), total=len(title_list),desc='计算title句子长度'))
title_l_list = [len(doc) for doc in docs]

title_sent_l_arr = np.array(title_l_list)
np.save(data_dir + 'c4-title句子长度数组.npy', title_sent_l_arr)

title_token_l_list = []

for sent in tqdm(title_list, desc='计算title句子token长度'):
    title_token_l_list.append(len(tokenizer.tokenize(sent)))

title_token_l_arr = np.array(title_token_l_list)
np.save(data_dir + 'c4-title句子token长度数组.npy', title_token_l_arr)

# 计算abstract
batch_size = 5000
docs = list(tqdm(nlp.pipe(abstrct_list, batch_size=batch_size, disable=["ner", "parser", "textcat"], n_process=10), total=len(abstrct_list),desc='计算abstract句子长度'))
abstrct_l_list = [len(doc) for doc in docs]

abstrct_sent_l_arr = np.array(abstrct_l_list)
np.save(data_dir + 'c4-abstrct句子长度数组.npy', abstrct_sent_l_arr)

abstrct_token_l_list = []

for sent in tqdm(abstrct_list, desc='计算title句子token长度'):
    abstrct_token_l_list.append(len(tokenizer.tokenize(sent)))

abstrct_token_l_arr = np.array(abstrct_token_l_list)
np.save(data_dir + 'c4-abstrct句子token长度数组.npy', abstrct_token_l_arr)