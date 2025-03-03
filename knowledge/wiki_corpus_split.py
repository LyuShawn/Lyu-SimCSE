from tqdm import tqdm
import argparse

# 取出所有句子，分句存在txt中
def main(args):


    domain = args.domain
    lang = args.lang
    output_file = args.output_file
    seed = args.seed
    multi_lang = args.multi_lang


    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Medicine")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output_file", type=str, default="data/wiki_corpus.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi_lang", action="store_true")
    args = parser.parse_args()
    main(args)
