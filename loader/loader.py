from utils.sentence_util import text_encode
import json
from knowledge.retrieval import retrieve_knowledge

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
        sent_features = {}
        prompt_prefix_input_ids = model_args.prompt_prefix_input_ids
        prompt_suffix_input_ids = model_args.prompt_suffix_input_ids

        if model_args.prompt_template2:
            prompt_prefix_input_ids2 = model_args.prompt_prefix_input_ids2
            prompt_suffix_input_ids2 = model_args.prompt_suffix_input_ids2
        else:
            prompt_prefix_input_ids2 = prompt_prefix_input_ids
            prompt_suffix_input_ids2 = prompt_suffix_input_ids

        eval_template = model_args.eval_template
        if eval_template:
            eval_prefix = eval_template.split("{sentence}")[0]
            eval_suffix = eval_template.split("{sentence}")[1]
            eval_prefix_input_ids = tokenizer(eval_prefix)['input_ids']
            eval_suffix_input_ids = tokenizer(eval_suffix)['input_ids']

        input_ids = []
        attention_mask = []
        for i,s in enumerate(sentences):

            knowledge_mark = "{knowledge}"
            if knowledge_mark in model_args.prompt_template:
                knowledge = retrieve_knowledge(s, 
                                            retrieve_type=args.model_args.knowledge_retrieve_type,
                                            max_length = args.model_args.knowledge_max_length)
                if knowledge:
                    template = model_args.prompt_template.replace(knowledge_mark, knowledge)
                else:
                    template = model_args.eval_template

                prompt_prefix = template.split("{sentence}")[0]
                prompt_suffix = template.split("{sentence}")[1]
                prompt_prefix_input_ids = tokenizer(prompt_prefix)['input_ids']
                prompt_suffix_input_ids = tokenizer(prompt_suffix)['input_ids']

            # 处理拼接input_ids
            if i < total:
                # 不处理对齐，直接拼接
                s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                if model_args.knowledge_fusion_type == "positive":
                    # 原句子做和prompt后的句子做正样例
                    input_ids.append(s['input_ids'])
                elif model_args.knowledge_fusion_type == "knowledge_positive":
                    # eval_template中的句子和融入的知识做正样例
                    input_ids.append(eval_prefix_input_ids + s['input_ids'] + eval_suffix_input_ids)
                else:
                    input_ids.append(prompt_prefix_input_ids + s['input_ids'] + prompt_suffix_input_ids)
            elif i < total*2:
                s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                input_ids.append(prompt_prefix_input_ids2 + s['input_ids'] + prompt_suffix_input_ids2)
            else:
                if not s:
                    input_ids.append([])
                else:
                    s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                    input_ids.append(eval_prefix_input_ids + s['input_ids'] + eval_suffix_input_ids)

            if model_args.mask_prompt:
                # mask prompt，prompt是0
                attention_mask.append([0] * len(prompt_prefix_input_ids) + s['attention_mask'] + [0] * len(prompt_suffix_input_ids))
            else:
                # 不mask prompt，全关注
                attention_mask.append([1] * len(input_ids[-1])) # 用刚刚拼接的input_ids长度

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
    if model_args.knowledge_hard_negative:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

    if model_args.do_knowledge_fusion:
        sent_knowledge_list = []
        # 如果需要知识融合，对每个原始句子做知识检索，并tokenize
        for sent in examples[sent0_cname]:
            knowledge = retrieve_knowledge(sent, args.model_args.knowledge_retrieve_type)
            sent_knowledge_list.append(knowledge if knowledge else "")
        sent_knowledge_features = tokenizer(
            sent_knowledge_list,
            max_length=256,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features['sent_knowledge'] = sent_knowledge_features['input_ids']

    return features