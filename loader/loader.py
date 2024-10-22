
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

    total = len(examples[sent0_cname])

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


    if model_args.do_prompt_enhancement:
        sent_features = {}
        model_args.prompt_prefix_input_ids = tokenizer(model_args.prompt_prefix)["input_ids"][:-1]
        model_args.prompt_suffix_input_ids = tokenizer(model_args.prompt_suffix)["input_ids"][1:]
        input_ids = []
        attention_mask = []
        for i,s in enumerate(sentences):
            s = tokenizer(s,max_length=data_args.max_seq_length,truncation=True,padding="max_length" if data_args.pad_to_max_length else False,)
            input_ids.append(model_args.prompt_prefix_input_ids + s['input_ids'] + model_args.prompt_suffix_input_ids)
            attention_mask.append([0] * len(model_args.prompt_prefix_input_ids) + s['attention_mask'] + [0] * len(model_args.prompt_suffix_input_ids))

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
        
    return features