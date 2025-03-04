from tqdm import tqdm
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from backend import MySQLClient
import logging
import jieba

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

nltk.download('punkt')
nltk.download('punkt_tab')
MySQL = MySQLClient()

lang_mapping = {
    'de': 'german',      # 德语
    'en': 'english',     # 英语
    'es': 'spanish',     # 西班牙语
    'fr': 'french',      # 法语
    'it': 'italian',     # 意大利语
    'nl': 'dutch',       # 荷兰语
    'pl': 'polish',      # 波兰语
    'pt': 'portuguese',  # 葡萄牙语
    'ru': 'russian',     # 俄语
    'zh': 'Chinese'      # 中文
}

# 取出所有句子，分句存在txt中
def main(args):


    domain = args.domain
    lang = args.lang
    output_file = args.output_file
    multi_lang = args.multi_lang

    offset = 0
    limit = 1000

    total = MySQL.get_wiki_page_content_multilingual_len(lang)

    pbar = tqdm(total=total, desc="collecting sentences")

    sent_list = []

    while True:
        page_content_dict = MySQL.get_wiki_page_content_multilingual(lang, offset) 
        if not page_content_dict:
            logging.info("no more page content")
            break
        pbar.update(len(page_content_dict))
        offset += limit
        for page_id, page_content in page_content_dict.items():
            if lang == 'zh':
                sent_list += sent_tokenize(page_content)
            else:
                sent_list += sent_tokenize(page_content, language=lang_mapping[lang])
        pbar.set_description_str(f"collecting sentences: {len(sent_list)}")

    logging.info(f"total sentences: {len(sent_list)}")
    logging.info(f"writing to file: {output_file}")
    with open(output_file, "w") as f:
        for sent in sent_list:
            f.write(sent + "\n")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Medicine")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output_file", type=str, default="data/wiki_corpus.txt")
    parser.add_argument("--multi_lang", action="store_true")
    args = parser.parse_args()
    main(args)
