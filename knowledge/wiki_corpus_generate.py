"""
生成wiki的相关语料
输入包括：（领域、行数、输出文件路径）
"""
import argparse
import os
from tqdm import tqdm
from backend import MySQLClient
import nltk
from nltk.tokenize import sent_tokenize
from transformers import set_seed
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
nltk.download('punkt')
MySQL = MySQLClient()

def main(args):
    domain = args.domain
    sent_num = args.sent_num
    output_file = args.output_file
    max_sent_num = args.max_sent_num
    seed = args.seed

    if seed:
        # 设置随机种子
        set_seed(seed)

    sent_list = []
    sent_cnt = 0
    offset = 0
    limit = 1000
    pbar = tqdm(total=max_sent_num, desc="collecting sentences")

    while True:
        page_content_dict = MySQL.get_wiki_page_content(domain, offset)    # page_id -> page_content
        if not page_content_dict:
            logging.info("no more page content")
            break
        pbar.set_postfix_str(f"pages: {offset}")
        offset += limit
        # 取出句子
        for page_id, page_content in page_content_dict.items():
            sent_list_tmp = sent_tokenize(page_content)
            sent_list += sent_list_tmp
            sent_cnt += len(sent_list_tmp)

            pbar.update(len(sent_list_tmp))

        if max_sent_num != -1 and sent_cnt >= max_sent_num:
            logging.info(f"max_sent_num: {max_sent_num} reached")
            break

    # 采样
    logging.info(f"sent_cnt: {sent_cnt}")
    if sent_num != -1 and sent_num < sent_cnt:
        sent_list = random.sample(sent_list, sent_num)
    # 写入文件
    logging.info(f"writing to {output_file}")
    with open(output_file, "w") as f:
        for sent in sent_list:
            f.write(sent + "\n")
    logging.info("done")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Medicine")
    parser.add_argument("--sent_num", type=int, default=1_000_000)  # -1时不采样
    parser.add_argument("--max_sent_num", type=int, default=-1) # 限制最大句子数
    parser.add_argument("--output_file", type=str, default="data/wiki_corpus.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
