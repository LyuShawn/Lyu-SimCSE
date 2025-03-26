from knowledge.backend import RedisClient, MySQLClient
from utils.sentence_util import text_encode
import json
from utils.cache_util import disk_cache
from utils.sentence_util import text_md5
import random
import string
import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()

def retrieval_knowledge_title(sent_list):
    """搜索句子对应的知识标题"""
    mysql = MySQLClient()
    result = []
    for sent in sent_list:
        sent_md5 = text_md5(sent)
        title_list = mysql.get_sent_page_in_by_md5(sent_md5)
        result.append(title_list)
    return result

def retrieval_knowledge_summary(sent_list,max_length = -1):
    redis_client = RedisClient()
    search_prifix = "wikisearch:"

    keys = [search_prifix + text_encode(sent) for sent in sent_list]
    values = redis_client.mget(keys)
    page_ids = []
    for value in values:
        if not value:
            page_ids.append(None)
            continue
        value = json.loads(value)
        if not value:
            page_ids.append(None)
            continue

        # 获取第一个pageid
        page_id = value[0]["page_id"]
        page_ids.append(page_id)
    page_prifix = "wikipage:"
    keys = [page_prifix + str(page_id) for page_id in page_ids]
    values = redis_client.mget(keys)
    result = []
    for value in values:
        if not value:
            result.append(None)
            continue
        summary = json.loads(value)["summary"]
        if max_length == -1:
            result.append(summary)
        else:
            summary = summary.split()[0:max_length]
            result.append(" ".join(summary))
    return result


def retrieval_knowledge_sentence(sent_list,max_length = -1):
        redis_client = RedisClient(db=2)
        prifix = "similarity_sent_"
        keys = [prifix + text_encode(sent) for sent in sent_list]
        values = redis_client.mget(keys)
        result = []
        if values:
            for value in values:
                if not value:
                    result.append([])
                    continue
                value = json.loads(value)
                result.append(value)
        return result

def retrieval_knowledge(sent_list, retrieve_type = 'title', max_length = -1):
    """
        查询知识
    """

    type_list = ["title","summary","empty","sentence","rewrite","random","random_char","random_word","unknown"]
    assert retrieve_type in type_list, f"retrieve_type must in {type_list}"

    result = []
    if retrieve_type=="title":
        knowledge_list = retrieval_knowledge_title(sent_list)
        # 先拼接再截断
        for knowledge in knowledge_list:
            if not knowledge:
                result.append("")
                continue
            knowledge = ",".join(knowledge)
            if max_length == -1:
                result.append(knowledge)
            else:
                knowledge = knowledge.split()[0:max_length]
                result.append(" ".join(knowledge))
        return result
    elif retrieve_type=="summary":
        return retrieval_knowledge_summary(sent_list,max_length)
    elif retrieve_type=="sentence":
        knowledge_list = retrieval_knowledge_sentence(sent_list,max_length)
        sim_threshold = 0.5

        for knowledge in knowledge_list:
            if not knowledge:
                result.append("")
                continue
            
            k_sent_list = [sent for sim, sent in knowledge if sim > sim_threshold]
            if not k_sent_list:
                result.append("")
                continue
            k_sent_list = ",".join(k_sent_list)
            result.append(k_sent_list)
        return result
    elif retrieve_type=="empty":
        knowledge_list = retrieval_knowledge_title(sent_list)
        for k in knowledge_list:
            if k:
                result.append("{knowledge}")
            else:
                result.append("")
        return result
    elif retrieve_type=="rewrite":
        return sent_list

    elif retrieve_type=="unknown":
        return ["unknown"]*len(sent_list)
    elif retrieve_type=="random_char":
        # 随机生成随机数量的字符
        result = []
        for sent in sent_list:
            k = random.randint(1,32)
            result.append(''.join(random.choices(string.ascii_letters, k=k)))
        return result
    elif retrieve_type=="random_word":
        # 随机选择随机数量的词
        result = []
        for sent in sent_list:
            length = random.randint(1,32)
            result.append(" ".join(random.sample(word_list,length)))
        return result
    else:
        raise NotImplementedError

