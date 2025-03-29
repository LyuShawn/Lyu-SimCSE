# 计算混合语言数据集的长度
import os
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

output_file = "draw-data/c5-混合语料长度.npz"
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
input_path = "../Dataset-LyuCSE/wiki1m_multi_lang.csv"
df = pd.read_csv(input_path)
sent_list = df['text'].tolist()

token_l_list = []
l_list = []
for sent in tqdm(sent_list, desc='计算句子长度'):
    doc = nlp(sent)
    l_list.append(len(doc))
    token_l_list.append(len(tokenizer.tokenize(sent)))

sent_l_arr = np.array(l_list)
token_l_arr = np.array(token_l_list)
result = {"sent_l":sent_l_arr,"token_l":token_l_arr}

np.savez(output_file, **result)
