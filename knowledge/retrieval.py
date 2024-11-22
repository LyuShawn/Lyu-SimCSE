from knowledge.backend import RedisClient
from utils.sentence_util import text_encode
import json

def retrieve_knowledge(sent, retrieve_type = 'title', max_length = -1):
    """
        文本明文，返回知识库中的知识明文
    """
    # 单例
    try:
        raise Exception("redis_client")
        if retrieve_type == 'title':
            redis_client = RedisClient()
            prifix = "wikisearch:"
            key = prifix + text_encode(sent)
            value = json.loads(redis_client.get(key))
            # 组装知识，把所有title拼接起来
            knowledge = ""
            for item in value:
                knowledge += item["title"] + ","
            return knowledge[:-1]  # 去掉最后的逗号

        if retrieve_type == 'summary':
            redis_client = RedisClient()
            search_prifix = "wikisearch:"

            key = search_prifix + text_encode(sent)
            value = json.loads(redis_client.get(key))
            # 获取第一个pageid
            page_id = value[0]["page_id"]
            page_prifix = "wikipage:"
            key = page_prifix + str(page_id)
            value = redis_client.get(key)
            summary = json.loads(value)["summary"]
            if max_length == -1:
                return summary
            else:
                summary = summary.split()[0:max_length]
                return " ".join(summary)

        if retrieve_type == 'sentence':
            redis_client = RedisClient(db=2)
            prifix = "similarity_sent_"
            key = prifix + text_encode(sent)
            value = json.loads(redis_client.get(key))
            return value

    except Exception as e:
        return None
