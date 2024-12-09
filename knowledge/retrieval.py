from knowledge.backend import RedisClient
from utils.sentence_util import text_encode
import json


def retrieval_knowledge_title(sent_list):
    redis_client = RedisClient()
    prifix = "wikisearch:"
    keys = [prifix + text_encode(sent) for sent in sent_list]
    values = redis_client.mget(keys)
    result = []
    for value in values:
        if not value:
            result.append([])
            continue
        value = json.loads(value)
        result.append([item["title"] for item in value])
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
                knowledge = knowledge[0:max_length]
                result.append(knowledge)
        return result
    elif retrieve_type=="summary":
        return retrieval_knowledge_summary(sent_list,max_length)
    elif retrieve_type=="sentence":
        return retrieval_knowledge_sentence(sent_list,max_length)
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
    else:
        raise NotImplementedError