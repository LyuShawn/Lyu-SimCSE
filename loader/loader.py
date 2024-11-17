from utils.sentence_util import text_encode
import redis
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

    # 如果有第三个句子
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    if "{title}" in model_args.prompt_template:
        r = redis.Redis(host='59.77.134.205', port=6379, db=0, password='lyuredis579')


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

        input_ids = []
        attention_mask = []
        for i,s in enumerate(sentences):

            if "{title}" in model_args.prompt_template:
                info = r.get(f"wikisearch:{text_encode(s)}")
                use_title = False
                if info:
                    info = json.loads(info)
                    if len(info) > 0:
                        title_list = [item['title'] for item in info]
                        if model_args.page_title_num != -1:
                            title_list = title_list[:model_args.page_title_num]
                        title_str = ", ".join(title_list)
                        template = model_args.prompt_template.replace("{title}", title_str)
                        use_title = True
                if not use_title:
                    template = model_args.eval_template

                prompt_prefix = template.split("{sentence}")[0]
                prompt_suffix = template.split("{sentence}")[1]
                prompt_prefix_input_ids = tokenizer(prompt_prefix)['input_ids']
                prompt_suffix_input_ids = tokenizer(prompt_suffix)['input_ids']

            # 处理拼接input_ids
            if i < total:
                # 不处理对齐，直接拼接
                s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                input_ids.append(prompt_prefix_input_ids + s['input_ids'] + prompt_suffix_input_ids)
            elif i < total*2:
                s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                input_ids.append(prompt_prefix_input_ids2 + s['input_ids'] + prompt_suffix_input_ids2)
            else:
                s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
                input_ids.append(prompt_prefix_input_ids2 + s['input_ids'] + prompt_suffix_input_ids2)

            if model_args.mask_prompt:
                # mask prompt，prompt是0
                attention_mask.append([0] * len(prompt_prefix_input_ids) + s['attention_mask'] + [0] * len(prompt_suffix_input_ids))
            else:
                # 不mask prompt，全关注
                attention_mask.append([1] * (len(prompt_prefix_input_ids) + len(s['input_ids']) + len(prompt_suffix_input_ids)))

        sent_features['input_ids'] = input_ids

        # 不做对齐，在collator里做batch内对齐
        # ml = max([len(i) for i in sent_features['input_ids']])
        # # 对齐
        # for i in range(len(sent_features['input_ids'])):
        #     t = sent_features['input_ids'][i]
        #     sent_features['input_ids'][i] = t + [tokenizer.pad_token_id]*(ml-len(t))
        #     if model_args.mask_prompt:
        #         # 按前面的attention_mask补0到ml
        #         attention_mask[i] = attention_mask[i] + [0] * (ml-len(attention_mask[i]))
        #     else:
        #         # 不用管prompt，全关注
        #         attention_mask.append(len(t)*[1] + (ml-len(t))*[0])

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

    if args.model_args.do_knowledge_fusion:
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