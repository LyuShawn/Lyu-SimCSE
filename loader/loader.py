from utils.sentence_util import text_encode
import json
from knowledge.retrieval import retrieve_knowledge,retrieval_knowledge_batch

class PrepareFeaturesArgs:
    def __init__(self, tokenizer, data_args, model_args, sent0_cname, sent1_cname, sent2_cname):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.sent0_cname = sent0_cname
        self.sent1_cname = sent1_cname
        self.sent2_cname = sent2_cname

def prepare_features(examples, args:PrepareFeaturesArgs):
    """
    处理生成句子特征
    """
    sent0_cname = args.sent0_cname
    sent1_cname = args.sent1_cname
    sent2_cname = args.sent2_cname
    tokenizer = args.tokenizer
    data_args = args.data_args
    model_args = args.model_args

    total = len(examples[sent0_cname])  # 1000

    # 避免空值
    for idx in range(total):
        if examples[args.sent0_cname][idx] is None:
            examples[args.sent0_cname][idx] = " "
        if examples[args.sent1_cname][idx] is None:
            examples[args.sent1_cname][idx] = " "

    sentences = examples[sent0_cname] + examples[sent1_cname]

    if model_args.knowledge_hard_negative:
        knowledge_hard_sent_list = []
        threshold = 0.9
        for s in examples[sent0_cname]:
            knowledge_sent_list = retrieve_knowledge(s, retrieve_type='sentence') # [(cos_sim, sent)]
            knowledge_hard_sent = ""
            if knowledge_sent_list:
                # 选择相似度小于0.9的最大的句子
                knowledge_sent_list = [(cos_sim, sent) for cos_sim, sent in knowledge_sent_list if cos_sim < threshold]
                if knowledge_sent_list:
                    knowledge_sent_list.sort(key=lambda x: x[0], reverse=True)
                    knowledge_hard_sent = knowledge_sent_list[0][1]
            knowledge_hard_sent_list.append(knowledge_hard_sent)
        sentences += knowledge_hard_sent_list

    # 如果有第三个句子
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    if model_args.do_prompt_enhancement:
        knowledge_mark = model_args.knowledge_mark
        sent_mark = model_args.sent_mark
        if knowledge_mark in model_args.prompt_template:
            knowledge_list = retrieval_knowledge_batch(examples[sent0_cname], retrieve_type=args.model_args.knowledge_retrieve_type,max_length=args.model_args.knowledge_max_length)
        sent_features = {}

        # 模板选择
        template = model_args.prompt_template
        eval_template = model_args.eval_template

        input_ids = []
        attention_mask = []
        for i,s in enumerate(sentences):

            assert sent_mark in template or sent_mark in eval_template, "prompt_template or eval_template must contain [sentence]"

            s = s.split()[:model_args.max_seq_length]   # 提前截断句子
            len_template = len(template.split()) + 1

            if model_args.knowledge_fusion:
                # 做知识融合
                knowledge = knowledge_list[i % total]
                # 组装知识文本
                if knowledge:
                    knowledge = ",".join(knowledge)
                    template = template.replace(knowledge_mark, knowledge)
                else:
                    template = eval_template

                type = model_args.knowledge_fusion
                if type == "positive":
                    sent0 = template.replace(sent_mark, s)
                    sent1 = sent0
                elif type == "knowledge_positive":
                    sent0 = template.replace(sent_mark, s)
                    sent1 = eval_template.replace(sent_mark, s)
                else:
                    raise NotImplementedError

            else:
                # prompt_bert
                sent0 = template.replace(sent_mark, s)
                sent1 = eval_template.replace(sent_mark, s)

            sent0_input = tokenizer.encode(sent0, max_length=data_args.max_seq_length,
                    truncation=True,
                    padding="max_length" if data_args.pad_to_max_length else False,)
            if sent0 != sent1:
                sent1_input = tokenizer.encode(sent1, max_length=data_args.max_seq_length,
                    truncation=True,
                    padding="max_length" if data_args.pad_to_max_length else False,)
            else:
                sent1_input = sent0_input

            if i < total:
                input_ids.append(sent0_input)
            elif i < total*2:
                input_ids.append(sent1_input)
            else:
                raise NotImplementedError
            
            if tokenizer.mask_token_id not in input_ids[-1]:
                a = 1
                pass
            assert tokenizer.mask_token_id in input_ids[-1], "mask token not in input_ids"

            attention_mask.append([1] * len(input_ids[-1]))

        sent_features['input_ids'] = input_ids
        sent_features['attention_mask'] = attention_mask

    else:
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

    # if model_args.do_knowledge_fusion:
    #     sent_knowledge_list = []
    #     # 如果需要知识融合，对每个原始句子做知识检索，并tokenize
    #     for sent in examples[sent0_cname]:
    #         knowledge = retrieve_knowledge(sent, args.model_args.knowledge_retrieve_type)
    #         sent_knowledge_list.append(knowledge if knowledge else "")
    #     sent_knowledge_features = tokenizer(
    #         sent_knowledge_list,
    #         max_length=256,
    #         truncation=True,
    #         padding="max_length" if data_args.pad_to_max_length else False,
    #     )

    #     features['sent_knowledge'] = sent_knowledge_features['input_ids']

    return features