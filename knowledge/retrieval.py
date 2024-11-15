from knowledge.backend import RedisClient
from utils.sentence_util import text_encode
import json

def retrieve_knowledge(sent):
    """
        文本明文，返回知识库中的知识明文
    """
    # 单例
    redis_client = RedisClient()

    prifix = "wikisearch:"
    key = prifix + text_encode(sent)

    value = redis_client.get(key)
    if not value:
        return None

    value = json.loads(value)

    # 组装知识，把所有title拼接起来
    knowledge = ""
    for item in value:
        knowledge += item["title"] + ","

    return knowledge[:-1]  # 去掉最后的逗号

