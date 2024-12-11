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
    cls.knowledge_sim = Similarity(temp=cls.model_args.knowledge_temp)

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
    hidden_dim = cls.config.hidden_size # hidden size of BERT/RoBERTa

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

    assert pooler_output.shape == (batch_size * num_sent, hidden_dim)  # 优雅的assert张量形状

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

    if cls.model_args.knowledge_loss_type == "info_nce":
        # 计算L2正则化作为知识抑制损失
        k_sim = cls.knowledge_sim(z1.unsqueeze(1), z2.unsqueeze(0))
        ksl = loss_fct(k_sim, labels)
        loss = loss + cls.model_args.knowledge_loss_weight * ksl

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

        # 都以pad填充
        new_input_ids = torch.full(
            (batch_size, seq_len + prompt_prefix.shape[0] + prompt_suffix.shape[0]),
            cls.pad_token_id,
            dtype=torch.long
        ).to(device)

        for i in range(batch_size):
            origin_sent = input_ids[i, :non_pad_indices[i]]  # 非 PAD 部分
            new_input = torch.cat([
                origin_sent[0:1],        # CLS
                prompt_prefix,           # 前缀
                origin_sent[1:-1],       # 原句内容（去掉首尾）
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