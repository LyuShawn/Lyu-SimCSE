import base64

def text_encode(text):
    # base64 编码
    return base64.b64encode(text.encode()).decode()
def text_decode(text):
    # base64 解码
    return base64.b64decode(text.encode()).decode()
