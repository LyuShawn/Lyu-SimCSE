import base64
import hashlib

def text_encode(text):
    # base64 编码
    return base64.b64encode(text.encode()).decode()
def text_decode(text):
    # base64 解码
    return base64.b64decode(text.encode()).decode()

def text_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()
