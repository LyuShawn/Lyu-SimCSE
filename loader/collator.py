import torch
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class OurDataCollatorWithPadding:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding: Union[bool, str, PaddingStrategy] = True, max_length: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, mlm: bool = True, mlm_probability: float = 0.15, do_mlm: bool = True, model_args=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.do_mlm = do_mlm
        self.model_args = model_args
        

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)

        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return

        if self.model_args.batch_inner_shuffle:
            import random
            random.shuffle(features)

        if self.model_args.do_knowledge_fusion:
            # 保存并移除sent_knowledge_intput_ids
            sent_knowledge_input_ids = [item.pop('sent_knowledge') for item in features]
            sent_knowledge_attention_mask = []
            # 对齐
            ml = max([len(i) for i in sent_knowledge_input_ids])
            l = len(sent_knowledge_input_ids)
            for i in range(l):
                # 先做attention_mask，再做input对齐
                sent_knowledge_attention_mask.append([1] * len(sent_knowledge_input_ids[i]) + [0] * (ml-len(sent_knowledge_input_ids[i])))
                sent_knowledge_input_ids[i] = sent_knowledge_input_ids[i] + [self.tokenizer.pad_token_id]*(ml-len(sent_knowledge_input_ids[i]))
            sent_knowledge_input_ids = torch.tensor(sent_knowledge_input_ids, dtype=torch.long)
            sent_knowledge_attention_mask = torch.tensor(sent_knowledge_attention_mask, dtype=torch.long)

        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        # batch = self.tokenizer.pad(
        #     flat_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )

        input_ids = [f['input_ids'] for f in flat_features]
        attention_mask = [f['attention_mask'] for f in flat_features]
        # 对齐
        ml = max([len(i) for i in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [self.tokenizer.pad_token_id]*(ml-len(input_ids[i]))
            attention_mask[i] = attention_mask[i] + [0] * (ml-len(attention_mask[i]))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.model_args.do_knowledge_fusion:
            batch['sent_knowledge'] = {'input_ids': sent_knowledge_input_ids, 'attention_mask': sent_knowledge_attention_mask}

        if self.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        for k in special_keys:
            if k in batch:
                batch[k] = batch[k].view(bs, num_sent, -1)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels