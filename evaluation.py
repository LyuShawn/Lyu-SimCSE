import sys
import os
import logging
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer,HfArgumentParser
import json
from arguments import ModelArguments,EvalArguments
from datetime import datetime
from simcse.models import Pooler
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

class EvaluationUtil:

    sts_task_list =[
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICKRelatedness",
    ]
    transfer_task_list =["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    dev_sts_task_list = ["STSBenchmark", "SICKRelatedness"]
    dev_transfer_task_list = transfer_task_list

    def __init__(self, path, model_args, task_set="sts", mode="test", dataset=None,dataset_name=None, *args ,**kwargs):
        """数据准备"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.model_args = model_args
        self.task_set = task_set

        self.print_table_switch = False if not kwargs.get("print_table", None) else kwargs.get("print_table")

        # Set up the tasks
        if self.task_set == "sts":
            self.tasks = self.sts_task_list
        elif self.task_set == "transfer":
            self.tasks = self.transfer_task_list
        elif self.task_set == "full":
            self.tasks = self.sts_task_list + self.transfer_task_list

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mode = mode

        self.pooler_type = model_args.pooler_type

        if hasattr(model_args, "do_prompt_enhancement"):
            self.pooler_type = "mask" if model_args.do_prompt_enhancement else self.pooler_type

        self.pooler = Pooler(self.pooler_type).to(self.device)

        self.local_model = os.path.exists(path)
        self.base_path = path

        self.path = [path]

        if self.local_model:
            for model_path in os.listdir(path):
                if model_path.startswith("checkpoint-"):
                    self.path.append(os.path.join(path, model_path))
            logging.info(f"load model from local, path:{path}")
        else:
            logging.info(f"load model from huggingface, path:{path}")

        # 评估参数
        if self.mode == "dev" or self.mode == "fasttest":
            # Fast mode
            self.params = self.prepare_params(kfold=5,optim="rmsprop",batch_size=128,tenacity=3,epoch_size=2)
        elif self.mode == "test":
            # Full mode
            self.params = self.prepare_params(kfold=10,optim="adam",batch_size=256,tenacity=5,epoch_size=4)
        else:
            raise NotImplementedError

    @classmethod
    def print_table(cls, task_names, scores):
        tb = PrettyTable()
        tb.field_names = task_names
        tb.add_row(scores)
        logging.info(tb)

    def eval(self):
        """评估入口"""
        logging.info(
            f"start evaluation {self.path},with pooler={self.pooler_type},task_set={self.task_set},mode={self.mode}"
        )

        eval_result = {
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "eval_scores": {},
            "eval_details": [],
        }        

        for path in self.path:

            model = AutoModel.from_pretrained(path).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(path)

            if self.dataset is not None:
                result = self.eval_by_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=self.dataset,
                    dataset_name=self.dataset_name,
                    mode = self.mode,
                )
                result['avg'] = result['spearman']
            else:
                result = self.eval_core(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=self.tasks,
                    params=self.params,
                    pooler=self.pooler,
                    model_args=self.model_args,
                )
                result = self.process_result(result, self.tasks, mode=self.mode, print_table_switch=self.print_table_switch)
            eval_result["eval_details"].append({
                "path": path,
                "result": result,
                "avg": result["avg"],
            })

        # scores 保存最好的结果
        best_result = max(eval_result["eval_details"], key=lambda x: x["avg"])
        eval_result["eval_scores"] = best_result["result"]
        # 计算时间，记录秒
        eval_result["time_cost"] = float((datetime.now() - datetime.strptime(eval_result["eval_time"], "%Y-%m-%d %H:%M:%S")).seconds)

        if self.local_model:
            score_file_path = os.path.join(self.base_path, "eval_scores.json")
            with open(score_file_path, "w") as f:
                json.dump(eval_result, f, indent=4, sort_keys=True)
            return eval_result, score_file_path
        else:
            return eval_result

    @classmethod
    def process_result(cls, results, tasks, mode="test",print_table_switch=False):
        """处理senteval的结果"""

        scores = {}
        if mode == "dev":
            # dev模式
            for task in tasks:
                score = 0.00
                if task in ["STSBenchmark", "SICKRelatedness"]:
                    score = results[task]["dev"]["spearman"][0]
                elif task in cls.transfer_task_list:
                    score = results[task]["devacc"]
                else:
                    raise NotImplementedError
                scores[task] = score
        elif mode == "test" or mode == "fasttest":
            # test模式
            for task in tasks:
                score = 0.00
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    score = results[task]["all"]["spearman"]["all"]
                elif task in ["STSBenchmark", "SICKRelatedness"]:
                    score = results[task]["test"]["spearman"][0]
                elif task in cls.transfer_task_list:
                    score = results[task]["acc"]
                else:
                    raise NotImplementedError
                scores[task] = score

        scores["avg"] = sum(scores.values()) / len(scores)

        scores["sts_avg"] = sum([scores[task] for task in tasks if task in cls.sts_task_list]) / len(cls.sts_task_list)
        scores["transfer_avg"] = sum([scores[task] for task in tasks if task in cls.transfer_task_list]) / len(cls.transfer_task_list)

        if print_table_switch:
            task_names = list(scores.keys())
            table_scores = ["%.4f" % (score * 100) for score in scores.values()]
            cls.print_table(task_names, table_scores)

        return scores

    @classmethod
    def eval_core(
        cls,
        model,
        tokenizer,
        tasks,
        params,
        pooler,
        device=None,
        model_args=None,
        use_pooler_output=False,
    ):
        """评估核心"""

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode(# The above code is a Python script that outputs three hash
                # symbols "
                "utf-8") for word in s] for s in batch]

            sentences = [" ".join(s) for s in batch]

            if model_args.do_prompt_enhancement and model_args.eval_template:
                template = model_args.eval_template.replace("[MASK]", tokenizer.mask_token)
                for i, s in enumerate(sentences):
                    if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                    sentences[i] = template.replace("{sentence}", s).strip()
                # sentences = [template.replace("{sentence}", s).replace("[MASK]", tokenizer.mask_token) for s in sentences]

            # Tokenization

            batch = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=max_length if max_length is not None else False,
            )

            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)

            # Get raw embeddings
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True)

            return pooler(attention_mask = batch['attention_mask'],
                        outputs = outputs,
                        input_ids = batch['input_ids'],
                        mask_token_id = tokenizer.mask_token_id,
                        use_pooler_output = use_pooler_output)

        results = {}

        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result

        return results

    @classmethod
    def dev_eval(cls,model,tokenizer,tasks,params):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def prepare(params, samples):
            return
        
        def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode(# The above code is a Python script that outputs three hash
                # symbols "
                "utf-8") for word in s] for s in batch]

            sentences = [" ".join(s) for s in batch]
            batch = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=max_length if max_length is not None else False,
            )
            for k in batch:
                batch[k] = batch[k].to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            return outputs.pooler_output
            
        results = {}
        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result
        return results

    @classmethod
    def prepare_params(cls, kfold,optim,batch_size,tenacity,epoch_size):
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": kfold}
        params["classifier"] = {
            "nhid": 0,
            "optim": optim,
            "batch_size": batch_size,
            "tenacity": tenacity,
            "epoch_size": epoch_size,
        }
        # params["seed"] = 1111

        params["similarity"]= lambda s1, s2: F.cosine_similarity(s1, s2,dim=-1).tolist()

        return params

    @classmethod
    def eval_by_dataset_core(cls, model, tokenizer,dataset, sent1_name, sent2_name, label_name, bs=64):
        """评估核心，与senteval不同的是，这里是自己控制数据集
        评估由自己写
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pooler = Pooler("cls").to(device)

        def sent_tokenize(examples):
            total = len(examples[sent1_name])
            sentences = examples[sent1_name] + examples[sent2_name]
            sent_features = tokenizer(sentences, return_tensors="pt", padding=True, truncation=False)
            features = {}
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            return features

        dataset_tokenize = dataset.map(sent_tokenize, 
                                    batched=True,
                                    load_from_cache_file=True)

        cos_sim_list = []
        for batch in dataset_tokenize.batch(bs):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            flat_input_ids = []
            flat_attention_mask = []
            for i in range(len(input_ids)):
                flat_input_ids.extend(input_ids[i])
                flat_attention_mask.extend(attention_mask[i])

            # 对齐
            ml = max([len(i) for i in flat_input_ids])
            for i in range(len(flat_input_ids)):
                flat_input_ids[i] = flat_input_ids[i] + [tokenizer.pad_token_id]*(ml-len(flat_input_ids[i]))
                flat_attention_mask[i] = flat_attention_mask[i] + [0] * (ml-len(flat_attention_mask[i]))
            input_ids = torch.tensor(flat_input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(flat_attention_mask, dtype=torch.long).to(device)

            with torch.no_grad():
                try:
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True, 
                        return_dict=True,
                        sent_emb=True)
                except:
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True, 
                        return_dict=True,)

            # (bs*2, hidden_size)
            pooler_output = pooler(attention_mask=attention_mask, outputs=outputs, input_ids=input_ids, mask_token_id=tokenizer.mask_token_id, use_pooler_output=True)
            pooler_output = pooler_output.view(-1, 2, pooler_output.size(-1)) # (bs, 2, hidden_size)

            z1 = pooler_output[:, 0]    # (bs, hidden_size)
            z2 = pooler_output[:, 1]    # (bs, hidden_size)
            
            cos_sim = F.cosine_similarity(z1, z2, dim=-1) # (bs,)
            cos_sim_list.extend(cos_sim.tolist())
        assert len(cos_sim_list) == len(dataset), f"cos_sim_list:{len(cos_sim_list)}, dataset:{len(dataset)}"

        # 计算spearman相关系数和pearson相关系数
        label_list = np.array(dataset[label_name])
        # label除5
        cos_sim_list = np.array(cos_sim_list)
        spearman_corr, _ = spearmanr(cos_sim_list, label_list)
        pearson_corr, _ = pearsonr(cos_sim_list, label_list)
        return {"spearman": spearman_corr, "pearson": pearson_corr}

    @classmethod
    def eval_by_dataset(cls, model,tokenizer,dataset, dataset_name, mode, bs=64):
        """自己控制数据集"""
        # 判断数据集名称切分
        if "stsb_multi_mt" in dataset_name:
            # 多语言stsb数据集，切出dev数据集
            if mode == "test":
                dataset = dataset["test"]
            elif mode == "dev":
                dataset = dataset["dev"]
            else:
                raise NotImplementedError
            sent1_name = "sentence1"
            sent2_name = "sentence2"
            label_name = "similarity_score"

        else:
            raise NotImplementedError

        # 按照batch_size处理数据
        return cls.eval_by_dataset_core(model,tokenizer , dataset, sent1_name, sent2_name, label_name, bs=bs)

        

def main():
    # 解析命令行参数
    parser = HfArgumentParser((EvalArguments, ModelArguments))
    eval_args, model_args = parser.parse_args_into_dataclasses()
    
    eval_util = EvaluationUtil(**eval_args.__dict__, model_args=model_args)

    result = eval_util.eval()
    print(result)


if __name__ == "__main__":
    main()