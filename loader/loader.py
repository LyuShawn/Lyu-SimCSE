

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

    if model_args.mask_embedding_sentence:
        bs = tokenizer.encode(model_args.mask_embedding_sentence_bs)[:-1]
        es = tokenizer.encode(model_args.mask_embedding_sentence_es)[1:] # remove cls or bos

        if model_args.mask_embedding_sentence_different_template:
            bs2 = tokenizer.encode(model_args.mask_embedding_sentence_different_template)[:-1]
            es2 = tokenizer.encode(model_args.mask_embedding_sentence_different_template)[1:]
        else:
            bs2, es2 = bs, es

        sent_features = {'input_ids': [], 'attention_mask': []}
        for i, s in enumerate(sentences):
            if len(s) > 500:
                length = len(s)
                a= 1
            if i < total:
                # 这里做了最大句子长度的截断
                s = tokenizer.encode(s, add_special_tokens=False,max_length=data_args.max_seq_length,truncation=True)
                sent_features['input_ids'].append(bs+s+es)
            elif i < 2*total:
                s = tokenizer.encode(s, add_special_tokens=False,max_length=data_args.max_seq_length,truncation=True)
                sent_features['input_ids'].append(bs2+s+es2)
            else:
                s = tokenizer.encode(s, add_special_tokens=False,max_length=data_args.max_seq_length,truncation=True)
                sent_features['input_ids'].append(bs2+s+es2)
        # 计算最大序列长度
        ml = max([len(i) for i in sent_features['input_ids']])
        for i in range(len(sent_features['input_ids'])):
            t = sent_features['input_ids'][i]
            # 填充到最大长度
            sent_features['input_ids'][i] = t + [tokenizer.pad_token_id]*(ml-len(t))
            # 记录哪些toekn是实际的词，只需要关注这些token
            sent_features['attention_mask'].append(len(t)*[1] + (ml-len(t))*[0])

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