import redis
import json
import pymysql

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # 将初始化参数作为实例的唯一标识
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[key] = instance
        return cls._instances[key]

class RedisClient(metaclass=SingletonMeta):

    def __init__(self, host='59.77.134.205', port=6379, db=0):
        self._connection = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
        )
        
    def get_connection(self):
        return self._connection

    def get(self, key, *args, **kwargs):
        value = self._connection.get(key, *args, **kwargs)
        return value.decode() if value else None


    def mget(self, keys, *args, **kwargs):
        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list or tuple")
        values = self._connection.mget(keys, *args, **kwargs)
        return [value.decode() if value else None for value in values]

    def set(self, key, value, *args, **kwargs):

        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value)

        return self._connection.set(key, value, *args, **kwargs)

    def mset(self, key_value_dict, *args, **kwargs):
        """
        批量设置多个键值对，支持 JSON 数据
        """
        if not isinstance(key_value_dict, dict):
            raise TypeError("key_value_dict must be a dictionary")

        # 将所有值转换为 JSON 字符串
        for key, value in key_value_dict.items():
            if isinstance(value, (dict, list, tuple)):
                key_value_dict[key] = json.dumps(value)

        # 批量设置键值对
        return self._connection.mset(key_value_dict)

    def check_keys_exist(self, keylist, *args, **kwargs):
        if not isinstance(keylist, (list, tuple)):
            raise TypeError("keylist must be a list or tuple")

        # 使用 exists 方法逐个检查键是否存在
        existing_keys = [key for key in keylist if self._connection.exists(key)]
        non_existing_keys = [key for key in keylist if not self._connection.exists(key)]

        return existing_keys, non_existing_keys

class MySQLClient(metaclass=SingletonMeta):
    """MySQL 客户端
    封装对 MySQL 数据库的操作，可以直接存取数据
    """
    def __init__(self):
        self.connection = pymysql.connect(
            host="59.77.134.205",
            user="root",
            password="lyumysql579",
            database="wiki"
        )

    def batch_set_wiki_page_content(self, page_content_dict, keyword,domain):
        """批量添加维基页面内容"""
        cursor = self.connection.cursor()
        sql = """
            INSERT INTO t_page_content (id, content, keyword, domain)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE id = id;
            """
        cursor.executemany(sql, [(page_id, page_content, keyword, domain) for page_id, page_content in page_content_dict.items()])
        self.connection.commit()
        cursor.close()

    def batch_get_wiki_page_content(self, page_id_list):
        """批量获取维基页面内容"""
        if not page_id_list:
            return {}
        cursor = self.connection.cursor()
        sql = """
            SELECT id, content FROM t_page_content WHERE id IN %s;
            """
        cursor.execute(sql, (page_id_list,))
        page_content_dict = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return page_content_dict

    def batch_page_content_id_exist(self, page_id_list):
        """批量检查维基页面内容是否存在
        返回存在的keylist和不存在的keylist
        """
        if not page_id_list:
            return [], []
        cursor = self.connection.cursor()
        sql = """
            SELECT id FROM t_page_content WHERE id IN %s;
            """
        cursor.execute(sql, (page_id_list,))
        existing_keys = [row[0] for row in cursor.fetchall()]
        non_existing_keys = list(set(page_id_list) - set(existing_keys))
        cursor.close()
        return existing_keys, non_existing_keys

    def get_wiki_page_content(self, domain, offset=0, limit=1000):
        """获取维基页面内容"""
        cursor = self.connection.cursor()
        sql = """
            SELECT id, content FROM t_page_content WHERE domain = %s LIMIT %s, %s;
            """
        cursor.execute(sql, (domain, offset, limit))
        page_content_dict = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return page_content_dict

    def insert_sent_page_in(self,sent, sent_base64, title, page_id, size, word_count, snippet):

        """插入sent和page相关的信息"""
        cursor = self.connection.cursor()
        sql = """
            INSERT INTO t_sent_page_in (sent, sent_base64, title, page_id, size, word_count, snippet)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE sent = sent;
            """
        cursor.execute(sql, (sent, sent_base64, title, page_id, size, word_count, snippet))
        self.connection.commit()
        cursor.close()

    def batch_insert_sent_page_in(self, sent_page_in_list):
        """批量插入sent和page相关的信息"""
        if not sent_page_in_list:
            return

        # 如果每个元素是dict而不是tuple，需要转换为tuple
        if isinstance(sent_page_in_list[0], dict):
            sent_page_in_list = [(item['sent'], item['sent_md5'], item['title'], item['page_id'], item['size'], item['word_count'], item['snippet']) for item in sent_page_in_list]

        cursor = self.connection.cursor()
        sql = """
            INSERT INTO t_sent_page_in (sent, sent_md5, title, page_id, size, word_count, snippet)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE sent = sent;
            """
        cursor.executemany(sql, sent_page_in_list)
        self.connection.commit()
        cursor.close()

    def batch_set_wiki_page_content_multilingual(self, page_content_dict, keyword, lang):
        """批量添加维基页面内容，多语言版本"""
        cursor = self.connection.cursor()
        sql = """
            INSERT INTO t_page_content_multilingual (page_id, content, keyword, lang)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE page_id = page_id;
            """
        cursor.executemany(sql, [(page_id, page_content, keyword, lang) for page_id, page_content in page_content_dict.items()])
        self.connection.commit()
        cursor.close()

    def batch_get_wiki_page_content_multilingual(self, page_id_list, lang):
        """批量获取维基页面内容"""
        if not page_id_list:
            return {}
        cursor = self.connection.cursor()
        sql = """
            SELECT id, content FROM t_page_content_multilingual WHERE id IN %s AND lang = %s;
            """
        cursor.execute(sql, (page_id_list,lang))
        page_content_dict = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return page_content_dict

    def batch_page_content_id_exist_multilingual(self, page_id_list, lang):
        """批量检查维基页面内容是否存在
        返回存在的keylist和不存在的keylist
        """
        if not page_id_list:
            return [], []
        cursor = self.connection.cursor()
        sql = """
            SELECT id FROM t_page_content_multilingual WHERE id IN %s AND lang = %s;
            """
        cursor.execute(sql, (page_id_list, lang))
        existing_keys = [row[0] for row in cursor.fetchall()]
        non_existing_keys = list(set(page_id_list) - set(existing_keys))
        cursor.close()
        return existing_keys, non_existing_keys