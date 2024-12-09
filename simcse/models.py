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
        mask_output = torch.where(empyt_knowledge_mask.unsqueeze(-1), origin_mask_output, mask_output)

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

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        """
        初始化交叉注意力机制层
        :param hidden_dim: 特征维度 (BERT 隐藏层大小)
        :param num_heads: 注意力头数
        :param dropout: 注意力的dropout比率
        """
        super(CrossAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence_output, knowledge_emb, empty_knowledge_mask):
        """
        前向传播
        :param sentence_output: (bs, seq_len, hidden_dim) 句子的 BERT 表征
        :param knowledge_emb: (bs, knowledge_num, hidden_dim) 句子相关的知识表征
        :param empty_knowledge_mask: (bs,) 是否不更新句子表征的掩码
        :return: (bs, seq_len, hidden_dim) 更新后的句子表征
        """
        bs, seq_len, hidden_dim = sentence_output.size()
        knowledge_num = knowledge_emb.size(1)

        # 转换维度以匹配 MultiheadAttention 的输入格式 (seq_len, bs, hidden_dim)
        sentence_output = sentence_output.transpose(0, 1)  # (seq_len, bs, hidden_dim)
        knowledge_emb = knowledge_emb.transpose(0, 1)      # (knowledge_num, bs, hidden_dim)

        # Query = 句子表征, Key/Value = 知识表征
        attn_output, _ = self.attention(query=sentence_output, key=knowledge_emb, value=knowledge_emb)

        # 残差连接 + LayerNorm
        sentence_output = self.layer_norm(sentence_output + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(sentence_output)
        updated_sentence_output = self.layer_norm(sentence_output + self.dropout(ffn_output))

        # 转回原始形状 (bs, seq_len, hidden_dim)
        updated_sentence_output = updated_sentence_output.transpose(0, 1)

        # 使用 empty_knowledge_mask 进行选择性更新
        # 如果 empty_knowledge_mask 为 True, 保持原始句子表征
        updated_sentence_output = torch.where(
            empty_knowledge_mask.unsqueeze(1).unsqueeze(2),  # (bs, 1, 1)
            sentence_output.transpose(0, 1),  # 原始句子表征 (bs, seq_len, hidden_dim)
            updated_sentence_output  # 更新后的表征
        )

        return updated_sentence_output

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

    pooler_type_list = ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last","mask","half_mask"]

    def __init__(self, pooler_type, **kwargs):
        super().__init__()

        if kwargs.get("do_prompt_enhancement"):
            if kwargs.get("knowledge_fusion_type") == "positive":
                self.pooler_type = "mask"
            else:
                self.pooler_type = "mask"
        else:
            self.pooler_type = pooler_type

        assert self.pooler_type in self.pooler_type_list, "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs, input_ids=None, mask_token_id=None, pooler_type=None,use_pooler_output=False):
        last_hidden = outputs.last_hidden_state # (bs, len, hidden)
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if use_pooler_output:
            return pooler_output

        pooler_type = pooler_type if pooler_type is not None else self.pooler_type
        assert pooler_type in self.pooler_type_list, "unrecognized pooling type %s" % pooler_type

        if pooler_type == "mask":

            assert input_ids is not None and mask_token_id is not None, "input_ids and mask_token_id should be provided for mask pooling"
            return last_hidden[input_ids == mask_token_id]

        if pooler_type == "half_mask":
            # last_hidden 前一半用cls，后一半用mask
            assert input_ids is not None and mask_token_id is not None, "input_ids and mask_token_id should be provided for mask pooling"
            bs = last_hidden.size(0) // 2
            # 全取，根据mask只会有一半有值
            cls_output = last_hidden[:bs, 0]    # (bs, hidden)
            mask_output = last_hidden[input_ids == mask_token_id]   # (bs, hidden)
            assert cls_output.size(0) == mask_output.size(0)
            return torch.cat([cls_output, mask_output], dim=0)

        if pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif pooler_type == "avg_top2":
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
    cls.pooler = Pooler(**cls.model_args.__dict__)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    # if cls.model_args.do_knowledge_fusion:
    #     # cls.knowledge_fusion = KnowledgeFussion(config.hidden_size, 
    #     #                                         cls.model_args).to(cls.device)
    #     cls.knowledge_fusion = CrossAttentionLayer(config.hidden_size, 
    #                                                 cls.model_args.knowledge_attention_head_num, 
    #                                                 cls.model_args.knowledge_attention_dropout).to(cls.device)

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
        
    pooler_output = cls.pooler(attention_mask, outputs, input_ids, cls.mask_token_id)

    assert pooler_output.size(0) == batch_size * num_sent and pooler_output.size(1) == outputs.last_hidden_state.size(-1)

    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    pooler_output = cls.mlp(pooler_output)

    if cls.model_args.do_knowledge_fusion:

        # 计算知识的表征，（bs,len,hidden）->(bs,hidden)
        if cls.knowledge_encoder is not None:
            knowledge_output = cls.knowledge_encoder(
                **sent_knowledge,
                return_dict=True,
            )
        else:
            knowledge_output = encoder(
                **sent_knowledge,
                return_dict=True,
            )

        # 对knowledge_output进行pooling
        # (bs, len, hidden) -> (bs, hidden)
        knowledge_output = cls.pooler(attention_mask = sent_knowledge["attention_mask"], 
                                    outputs = knowledge_output, 
                                    input_ids = sent_knowledge["input_ids"], 
                                    pooler_type = "mask",
                                    mask_token_id = cls.mask_token_id)
        if cls.model_args.knowledge_loss_type == "l2":
            # 计算L2正则化作为知识抑制损失
            ksl = torch.mean(torch.norm(knowledge_output, p=2, dim=-1) ** 2) * cls.model_args.knowledge_loss_weight
        else:
            raise NotImplementedError

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

    pooler_type = cls.pooler_type
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.do_prompt_enhancement:
        # input_ids: (bs, len)
        batch_size, seq_len = input_ids.shape

        device = input_ids.device

        prompt_prefix = torch.LongTensor(cls.eval_prefix_origin_input_ids).to(device)
        prompt_suffix = torch.LongTensor(cls.eval_suffix_origin_input_ids).to(device)

        mask = input_ids != cls.pad_token_id  # (batch_size, seq_len)
        non_pad_indices = mask.sum(dim=1)  # 每个句子的有效长度

        new_input_ids = torch.full(
            (batch_size, seq_len + prompt_prefix.shape[0] + prompt_suffix.shape[0]),
            cls.pad_token_id,
            dtype=torch.long
        ).to(device)

        for i in range(batch_size):
            origin_sent = input_ids[i, :non_pad_indices[i]]  # 非 PAD 部分
            new_input = torch.cat([
                prompt_prefix,           # CLS + 前缀
                origin_sent[:-1],        # 原句内容（去掉最后一个 token）
                prompt_suffix,           # 后缀
                origin_sent[-1:]         # 最后一个 token
            ])
            new_input_ids[i, :new_input.shape[0]] = new_input

        # 验证尺寸
        expected_length = seq_len + prompt_prefix.shape[0] + prompt_suffix.shape[0]
        assert new_input_ids.shape == (batch_size, expected_length)

        input_ids = new_input_ids
        attention_mask = (input_ids != cls.pad_token_id).long().to(device)
        token_type_ids = None
        pooler_type = "mask"

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

    pooler_output = cls.pooler(attention_mask, outputs, input_ids, cls.mask_token_id, pooler_type=pooler_type)
    
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

        if self.model_args.do_knowledge_fusion:
            if self.model_args.knowledge_encoder:
                self.knowledge_encoder = BertModel(config, add_pooling_layer=False)
            else:
                self.knowledge_encoder = None

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
                sent_knowledge=sent_knowledge,
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