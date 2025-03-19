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

    def get_sent_page_in_page_id_num(self):
        cursor = self.connection.cursor()
        sql = """
            SELECT count(DISTINCT page_id) FROM t_sent_page_in;
            """
        cursor.execute(sql)
        page_id_num = cursor.fetchall()
        page_id_num = page_id_num[0][0]
        cursor.close()
        return page_id_num

    def batch_get_sent_page_in_page_id(self, offset,limit=1000):
        """只拿pageid_list, page_id去重"""
        cursor = self.connection.cursor()
        sql = """
            SELECT DISTINCT page_id FROM t_sent_page_in LIMIT %s, %s;
            """
        cursor.execute(sql, (offset, limit))
        page_id_list = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return page_id_list

    def batch_get_sent_page_in_page_id_random(self,limit=1000):
        """只拿pageid_list, page_id去重"""
        cursor = self.connection.cursor()
        sql = """
            SELECT DISTINCT id,page_id
            FROM t_sent_page_in
            WHERE id >=
                (SELECT FLOOR(RAND() * (SELECT MAX(id) FROM t_sent_page_in)))
            ORDER BY id
            LIMIT %s;
            """
        cursor.execute(sql, (limit,))
        page_id_list = [row[1] for row in cursor.fetchall()]
        cursor.close()
        return page_id_list


    def get_sent_page_in_by_md5(self, sent_md5):
        """返回list"""
        if not sent_md5:
            return None
        sql = """
            SELECT title FROM t_sent_page_in WHERE sent_md5 = %s;
            """
        cursor = self.connection.cursor()
        cursor.execute(sql, (sent_md5,))
        title_list = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return title_list


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

    def get_wiki_page_content_multilingual(self, lang, offset=0, limit=1000):
        """获取维基页面内容"""
        cursor = self.connection.cursor()
        sql = """
            SELECT id, content FROM t_page_content_multilingual WHERE lang = %s LIMIT %s, %s;
            """
        cursor.execute(sql, (lang, offset, limit))
        page_content_dict = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return page_content_dict

    def get_wiki_page_content_multilingual_len(self, lang):
        """获取维基页面内容"""
        cursor = self.connection.cursor()
        sql = """
            SELECT count(id) FROM t_page_content_multilingual WHERE lang = %s;
            """
        cursor.execute(sql, (lang))
        page_content_len = cursor.fetchall()
        page_content_len = page_content_len[0][0]
        cursor.close()
        return page_content_len

    def get_wiki_page_content_len(self, domain):
        """获取维基页面内容"""
        cursor = self.connection.cursor()
        sql = """
            SELECT count(id) FROM t_page_content WHERE domain = %s;
            """
        cursor.execute(sql, (domain))
        page_content_len = cursor.fetchall()
        page_content_len = page_content_len[0][0]
        cursor.close()
        return page_content_len

    def get_page_info_size(self):
        cursor = self.connection.cursor()
        sql = """
            SELECT count(page_id) FROM t_page_info;
            """
        cursor.execute(sql)
        page_info_size = cursor.fetchall()
        page_info_size = page_info_size[0][0]
        cursor.close()
        return page_info_size

    def page_info_exist(self, page_id_list, lang):
        """检查page_id是否存在"""
        if not page_id_list:
            return [], []
        cursor = self.connection.cursor()
        sql = """
            SELECT page_id FROM t_page_info WHERE page_id IN %s AND lang = %s;
            """
        cursor.execute(sql, (page_id_list, lang))
        existing_keys = [row[0] for row in cursor.fetchall()]
        non_existing_keys = list(set(page_id_list) - set(existing_keys))
        cursor.close()
        return existing_keys, non_existing_keys

    def batch_insert_page_info(self, page_info_list,lang):
        """批量插入page相关的信息"""
        if not page_info_list:
            return

        # 如果每个元素是dict而不是tuple，需要转换为tuple
        if isinstance(page_info_list[0], dict):
            page_info_list = [(item['page_id'], item['title'], item['full_url'], item['categories'],lang, item['abstract']) for item in page_info_list]

        cursor = self.connection.cursor()
        sql = """
            INSERT INTO t_page_info (page_id, title, full_url, categories, lang, abstract)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE page_id = page_id;
            """
        cursor.executemany(sql,page_info_list) 
        self.connection.commit()
        cursor.close()

    def get_category_by_md5(self, sent_md5):
        """根据单个sent_md5获取category
        """
        cursor = self.connection.cursor()
        sql = """
            SELECT pi.categories
            FROM t_page_info pi
            JOIN (
                SELECT page_id
                FROM t_sent_page_in
                WHERE sent_md5 = %s
                ORDER BY id
                LIMIT 1
            ) sp ON pi.page_id = sp.page_id;
        """
        cursor.execute(sql, (sent_md5,))  # 传入单个sent_md5
        category_list = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return category_list[0] if category_list else None

def main():
    test_md5 = "ea4da97a4fae5c41e0f764431ae3d35b"
    mysql = MySQLClient()
    category = mysql.get_category_by_md5([test_md5])

if __name__ == "__main__":
    main()