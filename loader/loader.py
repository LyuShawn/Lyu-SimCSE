from knowledge.retrieval import retrieval_knowledge
from knowledge.prompt import get_random_prompt
from knowledge.backend import MySQLClient
from utils.sentence_util import text_md5

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

    features = {}

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
                                                retrieve_type=args.model_args.knowledge_retrieve_type)

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
                    # 截断knowledge
                    knowledge_tokens = tokenizer.tokenize(knowledge)[0:args.model_args.knowledge_max_length]
                    knowledge = tokenizer.convert_tokens_to_string(knowledge_tokens)
                    template = prompt_template.format(knowledge=knowledge,sentence='{sentence}')
                else:
                    template = model_args.eval_template

                prompt_prefix = template.split('{sentence}')[0]
                prompt_suffix = template.split('{sentence}')[1]
                prompt_prefix_input_ids = tokenizer.encode(prompt_prefix,truncation=True)[:-1] 
                prompt_suffix_input_ids = tokenizer.encode(prompt_suffix,truncation=True)[1:]

                if i < total:
                    # 不处理对齐，直接拼接
                    if model_args.knowledge_fusion_type == "knowledge_positive":
                        # eval_template中的句子和融入的知识做正样例
                        ii = eval_prefix_input_ids + s + eval_suffix_input_ids
                    elif model_args.knowledge_fusion_type == "self_positive":
                        ii = prompt_prefix_input_ids + s + prompt_suffix_input_ids
                    else:
                        raise NotImplementedError
                elif i < total*2:
                    ii = prompt_prefix_input_ids + s + prompt_suffix_input_ids
                else:
                    raise NotImplementedError
                if tokenizer.mask_token_id not in ii:
                    raise Exception("prompt_suffix_input_ids should contain mask token")
                input_ids.append(ii)

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

    else:
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

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

    if model_args.category_label:
        MySQL = MySQLClient()
        sent_category_list = []
        # 获取每个句子的category
        for sent in examples[sent0_cname]:
            sent_md5 = text_md5(sent)
            category_str = MySQL.get_category_by_md5(sent_md5)
            if not category_str:
                sent_category_list.append(['unknown'])
            else:
                category = category_str.replace('Category:', '')
                category = category.split('|')
                sent_category_list.append(category)
        # sent_category_list 是每个句子的category
        # token化
        category_feature_input_ids = []
        num_list = [len(i) for i in sent_category_list]
        # 拉平
        sent_category_list = [i for j in sent_category_list for i in j]
        category_features = tokenizer(sent_category_list, padding=False, truncation=True, max_length=32,add_special_tokens=False)
        # 重新分组
        start = 0
        for num in num_list:
            category_feature_input_ids.append(category_features['input_ids'][start:start+num])
            start += num
        features['category_input_ids'] = category_feature_input_ids

    return features