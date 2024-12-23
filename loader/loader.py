from knowledge.retrieval import retrieval_knowledge
from knowledge.prompt import get_random_prompt

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

    if model_args.knowledge_hard_negative:
        from knowledge.retrieval import retrieval_knowledge_sentence
        import random
        list = retrieval_knowledge_sentence(examples[sent0_cname])
        num = len(examples[sent0_cname])
        sent_list = []
        for item in list:
            if item:
                sent_list+=[sent for cos_sim,sent in item if cos_sim > 0.5]
        k_sent = random.sample(sent_list,min(num,len(sent_list)))

        if len(k_sent) < num:
            k_sent += ["N/A"]*(num-len(k_sent))

        assert len(k_sent) == num
        sentences += k_sent

    if model_args.do_prompt_enhancement:
        sent_features = {}

        prompt_prefix_input_ids = tokenizer.encode(model_args.prompt_prefix)[:-1]
        prompt_suffix_input_ids = tokenizer.encode(model_args.prompt_suffix)[1:]

        eval_template = model_args.eval_template
        eval_prefix = eval_template.split("{sentence}")[0]
        eval_suffix = eval_template.split("{sentence}")[1]
        eval_prefix_input_ids = tokenizer.encode(eval_prefix)[:-1]
        eval_suffix_input_ids = tokenizer.encode(eval_suffix)[1:]

        if model_args.knowledge_enhancement:
            knowledge_list = retrieval_knowledge(examples[sent0_cname], 
                                                retrieve_type=args.model_args.knowledge_retrieve_type,
                                                max_length=data_args.max_seq_length)

        input_ids = []
        attention_mask = []
        for i,s in enumerate(sentences):
            # sent做encode
            sent = s
            s = tokenizer.encode(s, add_special_tokens=False,
                    max_length=data_args.max_seq_length,
                    truncation=True,
                    padding="max_length" if data_args.pad_to_max_length else False,)

            if model_args.knowledge_enhancement:

                if model_args.random_prompt:
                    prompt_template = get_random_prompt()
                    prompt_template = prompt_template.replace("[MASK]", tokenizer.mask_token)
                else:
                    prompt_template = model_args.prompt_template

                knowledge = knowledge_list[i % total]
                if knowledge:
                    template = prompt_template.format(knowledge=knowledge,sentence='{sentence}')
                else:
                    template = model_args.eval_template

                prompt_prefix = template.split('{sentence}')[0]
                prompt_suffix = template.split('{sentence}')[1]
                prompt_prefix_input_ids = tokenizer.encode(prompt_prefix)[:-1]
                prompt_suffix_input_ids = tokenizer.encode(prompt_suffix)[1:]

                if i < total:
                    # 不处理对齐，直接拼接
                    if model_args.knowledge_fusion_type == "knowledge_positive":
                        # eval_template中的句子和融入的知识做正样例
                        input_ids.append(eval_prefix_input_ids + s + eval_suffix_input_ids)
                    elif model_args.knowledge_fusion_type == "self_positive":
                        input_ids.append(prompt_prefix_input_ids + s + prompt_suffix_input_ids)
                    else:
                        raise NotImplementedError
                elif i < total*2:
                    input_ids.append(prompt_prefix_input_ids + s + prompt_suffix_input_ids)
                else:
                    raise NotImplementedError

            else:
                # prompt_bert
                if i < total:
                    input_ids.append(prompt_prefix_input_ids + s + prompt_suffix_input_ids)
                elif i < total*2:
                    input_ids.append(eval_prefix_input_ids + s + eval_suffix_input_ids)
                else:
                    raise NotImplementedError

            attention_mask.append([1] * len(input_ids[-1]))

        sent_features['input_ids'] = input_ids
        sent_features['attention_mask'] = attention_mask
        # 对齐
        # sent_features['attention_mask'] = []
        # ml = max([len(i) for i in sent_features['input_ids']])
        # for i in range(len(sent_features['input_ids'])):
        #     t = sent_features['input_ids'][i]
        #     sent_features['input_ids'][i] = t + [tokenizer.pad_token_id]*(ml-len(t))
        #     sent_features['attention_mask'].append(len(t)*[1] + (ml-len(t))*[0])

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

    if model_args.knowledge_loss_type in ["k1_info_nce","k2_info_nce"]:
        # 处理knowledge_list
        template = model_args.eval_template
        knowledge_list = [template.format(sentence=knowledge) for knowledge in knowledge_list]
        # 如果需要知识融合，对每个原始句子做知识检索，并tokenize
        sent_knowledge_features = tokenizer(knowledge_list)
        features['sent_knowledge'] = sent_knowledge_features['input_ids']

    return features