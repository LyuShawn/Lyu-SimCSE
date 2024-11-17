import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class KnowledgeFussion(nn.Module):
    def __init__(self, hidden_dim, model_args):
        super(KnowledgeFussion, self).__init__()

        self.model_args = model_args

        self.attention = nn.MultiheadAttention(hidden_dim, 
                                                num_heads=model_args.knowledge_attention_head_num, 
                                                dropout=model_args.knowledge_attention_dropout)

        if model_args.freeze_attention_strength:
            self.attention_strength = model_args.knowledge_attention_strength
        else:
            self.attention_strength = nn.Parameter(torch.tensor(model_args.knowledge_attention_strength))


    def forward(self, model_output, 
                input_ids , 
                knowledge_output, 
                empyt_knowledge_mask,
                knowledge_token_id,
                mask_token_id):

        # model_output: (bs, len, hidden)
        # knowledge_output: (bs, hidden)
        bs, sent_len, hidden_dim = model_output.shape

        assert knowledge_output.shape == (bs, hidden_dim)

        origin_output = model_output.clone()    # (bs, len, hidden)

        knowledge_mask = input_ids == knowledge_token_id # (bs, len)
        
        knowledge_output = knowledge_output.unsqueeze(1).repeat(1, sent_len, 1) # (bs, len, hidden)
        # 根据knowledge_mask，替换掉model_output中的知识部分
        model_output[knowledge_mask] = knowledge_output[knowledge_mask]

        q = model_output[input_ids == mask_token_id] # (bs, hidden)
        # k = knowledge_output    # (bs, hidden)
        # v = model_output   # (bs, len, hidden)

        q = q.unsqueeze(1)  # (bs, 1, hidden)
        # k = k.unsqueeze(1)  # (bs, 1, hidden)

        k = model_output # (bs, len, hidden)
        v = model_output # (bs, len, hidden)

        q = q.transpose(0, 1)  # (1, bs, hidden)
        k = k.transpose(0, 1)  # (len, bs, hidden)
        v = v.transpose(0, 1)  # (len, bs, hidden)

        # attn_output: (len, bs, hidden)
        attn_output, attn_weights = self.attention(q, k, v)  # (len, bs, hidden)        

        attn_output = attn_output.transpose(0, 1)  # (bs, len, hidden)

        attn_output = attn_output.squeeze(1)  # (bs, hidden)

        # 调整模型输出
        mask_output = origin_output[input_ids == mask_token_id] # (bs, hidden)

        origin_mask_output = mask_output.clone()
        mask_output = mask_output + self.attention_strength * attn_output

        # 如果没有知识的部分，直接保留原始的 model_output（无变化）
        mask_output[empyt_knowledge_mask] = origin_mask_output[empyt_knowledge_mask]

        return mask_output  # (bs, hidden)

        # # 计算注意力更新
        # # 为了符合 multi-head attention 的输入要求，需要对模型输出进行转置
        # model_output = model_output.transpose(0, 1)  # 变为 (len, bs, hidden)
        
        # # 注意力机制：进行 self-attention 更新
        # attn_output, attn_weights = self.attention(model_output, model_output, model_output)  # (len, bs, hidden)
        
        # # # Step 3: 更新 model_output
        # attn_output = attn_output.transpose(0, 1)  # 转回 (bs, len, hidden)
        # model_output = model_output.transpose(0, 1)  # 转回 (bs, len, hidden)

        # # 使用注意力权重来调整模型输出
        # model_output = model_output + self.attention_strength * attn_output

        # # 如果没有知识的部分，直接保留原始的 model_output（无变化）
        # empyt_knowledge_mask = empyt_knowledge_mask.repeat(2) # (bs * len)
        # model_output[empyt_knowledge_mask] = origin_output[empyt_knowledge_mask]
        
        return model_output[input_ids == mask_token_id] # (bs, hidden)

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    if cls.model_args.do_knowledge_fusion:
        cls.knowledge_fusion = KnowledgeFussion(config.hidden_size, 
                                                cls.model_args).to(cls.device)

    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    sent_knowledge=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids   # (bs, num_sent, len)
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings    (bs * num_sent, len, hidden)
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling

    if cls.model_args.do_prompt_enhancement:
        if cls.model_args.do_knowledge_fusion:
            # 计算知识的表征，（bs,len,hidden）->(bs,hidden)
            knowledge_output = encoder(
                **sent_knowledge,
                return_dict=True,
            )
            # 对knowledge_output进行pooling
            knowledge_output = knowledge_output.last_hidden_state[:, 0, :]  # (bs, hidden)

            # 计算empty_knowledge_mask
            non_zero_count = torch.count_nonzero(sent_knowledge["input_ids"], dim=-1) # (bs)
            empty_knowledge_mask = non_zero_count <= 2

            last_hidden_state = outputs.last_hidden_state.view(batch_size, num_sent, -1, outputs.last_hidden_state.size(-1)) # (bs, num_sent, len, hidden)
            # 取一半
            last_hidden_state = last_hidden_state[:,1,:]    # (bs, len, hidden)

            # 原ori_input_ids取一半
            sent_input_ids = ori_input_ids[:,0,:]    # (bs, len)
            # 进行知识融合  (bs, hidden)
            mask_attn_output = cls.knowledge_fusion(
                model_output=last_hidden_state,
                input_ids=sent_input_ids,    # (bs , len)
                knowledge_output = knowledge_output, # (bs, hidden)
                empyt_knowledge_mask = empty_knowledge_mask, # (bs)
                knowledge_token_id = cls.model_args.knowledge_token_id,
                mask_token_id = cls.model_args.mask_token_id)

        pooler_output = outputs.last_hidden_state[input_ids == cls.model_args.mask_token_id].view(batch_size * num_sent, -1)
    else:
        pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    if cls.model_args.do_prompt_enhancement:
        pass
    else:
        if cls.pooler_type == "cls":
            pooler_output = cls.mlp(pooler_output)

    if cls.model_args.do_knowledge_fusion:
        if cls.model_args.knowledge_fusion_type == "full":
            raise NotImplementedError
            # z1, z2 = knowledge_pooler_output[:,0], knowledge_pooler_output[:,1]
        elif cls.model_args.knowledge_fusion_type == "selective":
            z1, z2 = pooler_output[:,0], mask_attn_output
        elif cls.model_args.knowledge_fusion_type == "fusion_loss":
            z1, z2 = pooler_output[:,0], pooler_output[:,1]
            z3 = mask_attn_output
            # z1, z2 = pooler_output[:,0], pooler_output[:,1]
    else:
        z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    if cls.model_args.do_knowledge_fusion and cls.model_args.knowledge_fusion_type == "fusion_loss":
        knowledge_sim = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        fusion_loss = loss_fct(knowledge_sim, labels)
        loss = loss + cls.model_args.knowledge_loss_weight * fusion_loss

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.do_prompt_enhancement:

        # 只做表征的时候，不会用到外部知识，所以不需要使用外部知识的prompt
        prompt_prefix_input_ids = torch.tensor(cls.model_args.eval_prefix_input_ids).to(input_ids.device)
        prompt_suffix_input_ids = torch.tensor(cls.model_args.eval_suffix_input_ids).to(input_ids.device)
        # TODO: 这里的拼接方式可能需要调整
        input_ids = torch.cat([prompt_prefix_input_ids.unsqueeze(0).expand(input_ids.size(0), -1), 
                            input_ids, 
                            prompt_suffix_input_ids.unsqueeze(0).expand(input_ids.size(0), -1)], dim=1)

        if cls.model_args.mask_prompt:
            # 拼接出新的attention_mask，将prompt部分的attention_mask设置为1，prefix和suffix部分设置为0
            attention_mask = torch.cat([torch.zeros(input_ids.size(0), prompt_prefix_input_ids.size(0)).to(input_ids.device), 
                                        attention_mask,
                                        torch.zeros(input_ids.size(0), prompt_suffix_input_ids.size(0)).to(input_ids.device)], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)

        token_type_ids = None
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    if cls.model_args.do_prompt_enhancement:
        # 这里会出现bs*2个true，原因是在外面已经拼接了prefix和suffix，这里又拼接了一次
        mask = input_ids == cls.model_args.mask_token_id
        pooler_output = outputs.last_hidden_state[mask]    # (bs, hidden)
        pooler_output = pooler_output.view(input_ids.shape[0], -1, pooler_output.shape[-1]).mean(1)
    else:
        pooler_output = cls.pooler(attention_mask, outputs)
        if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
            pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        sent_knowledge=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                sent_knowledge=sent_knowledge
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )