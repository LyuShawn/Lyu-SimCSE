import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize

# 下载nltk的 punkt 资源，用于句子分割
nltk.download('punkt')

# 初始化 Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('en')

def fetch_sentences(keyword, sentence_count):
    # 根据关键词搜索维基百科页面
    page = wiki_wiki.page(keyword)

    # 检查页面是否存在
    if not page.exists():
        print(f"页面 '{keyword}' 不存在！")
        return []

    # 获取页面的文本内容
    text = page.text

    # 将文本按句子进行分割
    sentences = sent_tokenize(text)

    # 如果请求的句子数量大于实际句子数，返回所有句子
    if len(sentences) < sentence_count:
        print(f"请求的句子数量 {sentence_count} 超过了实际可用句子的数量 {len(sentences)}")
        sentence_count = len(sentences)

    # 获取所需数量的句子
    selected_sentences = sentences[:sentence_count]

    return selected_sentences

def save_sentences_to_file(sentences, filename):
    # 将句子保存到文件
    with open(filename, 'w') as f:
        for sentence in sentences:
            f.write(sentence + "\n")

def main():
    # 用户输入关键词和需求的句子数量
    keyword = "medical"
    sentence_count = int(input("请输入你需要的句子数量: "))

    # 获取并保存句子
    sentences = fetch_sentences(keyword, sentence_count)
    if sentences:
        filename = f"{keyword}_sentences.txt"
        save_sentences_to_file(sentences, filename)
        print(f"已将 {len(sentences)} 个句子保存到 '{filename}' 文件中。")

if __name__ == "__main__":
    main()