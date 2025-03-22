# 评估
from transformers import AutoTokenizer,AutoModel
from datasets import load_dataset
import torch
from evaluation import EvaluationUtil
import json

model_name = "model/msimcse-xlm-roberta-large-cross_all/"

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
model = AutoModel.from_pretrained(model_name)

dataset_name = "mteb/stsb_multi_mt"
dataset = load_dataset(dataset_name, name="default")

result = EvaluationUtil.eval_by_dataset(model=model, tokenizer=tokenizer, dataset=dataset, dataset_name="Tatoeba.36", mode="test", bs=128,use_mteb=True)

output_file = 'tmp_result.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)
print(f"Result has been saved to {output_file}")
print(f"avg:{result['avg']}")
