import sys
import os
import logging
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer,HfArgumentParser
import json
from arguments import ModelArguments,EvalArguments
from datetime import datetime
import random
from simcse.models import BertForCL

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

eval_task_list =[
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICKRelatedness",
]

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    logging.info(tb)

class EvaluationUtil:
    def __init__(self, path, model_args, task_set="sts", mode="test", *args ,**kwargs):
        """数据准备"""

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.model_args = model_args
        self.task_set = task_set
        self.mode = mode
        self.pooler = model_args.pooler_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def eval(self):
        """评估入口"""
        logging.info(
            f"start evaluation {self.path},with pooler={self.pooler},task_set={self.task_set},mode={self.mode}"
        )

        eval_result = {
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_scores": {},
            "eval_details": [],
        }        

        for path in self.path:

            if self.local_model:
                model = BertForCL.from_pretrained(path, model_args=self.model_args)
            else:
                model = AutoModel.from_pretrained(path).to(self.device)

            tokenizer = AutoTokenizer.from_pretrained(path)
            model.to(self.device)

            # 只做文本表征时用
            args = {"sent_emb": True}            

            result = self.eval_core(
                model=model,
                tokenizer=tokenizer,
                args=args
            )
            # 处理结果
            result = self.process_result(result)
            eval_result["eval_details"].append({
                "path": path,
                "result": result,
                "avg": result["avg"],
            })

        # avg_scores 保存最好的结果
        best_result = max(eval_result["eval_details"], key=lambda x: x["avg"])
        eval_result["avg_scores"] = best_result["result"]

        if self.local_model:
            score_file_path = os.path.join(self.base_path, "avg_scores.json")
            with open(score_file_path, "w") as f:
                json.dump(eval_result, f, indent=4, sort_keys=True)
            return eval_result, score_file_path
        else:
            return eval_result

    def process_result(self, result):
        """处理senteval的结果"""
        task_scores = {}

        for task in eval_task_list:
            if task in result:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    task_scores[task] = result[task]["all"]["spearman"]["all"]
                else:
                    task_scores[task] = result[task]["test"]["spearman"].correlation

        task_scores["avg"] = sum(task_scores.values()) / len(task_scores)
        return task_scores

    def eval_core(
        self,
        model,
        tokenizer,
        args=None,
    ):
        """评估核心"""

        pooler = self.pooler

        # Set up the tasks
        if self.task_set == "sts":
            tasks = eval_task_list
        elif self.task_set == "transfer":
            tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
        elif self.task_set == "full":
            tasks = eval_task_list
            tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

        # Set params for SentEval
        if self.mode == "dev" or self.mode == "fasttest":
            # Fast mode
            params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
            params["classifier"] = {
                "nhid": 0,
                "optim": "rmsprop",
                "batch_size": 128,
                "tenacity": 3,
                "epoch_size": 2,
            }
        elif self.mode == "test":
            # Full mode
            params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
            params["classifier"] = {
                "nhid": 0,
                "optim": "adam",
                "batch_size": 64,
                "tenacity": 5,
                "epoch_size": 4,
            }
        else:
            raise NotImplementedError

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode("utf-8") for word in s] for s in batch]

            sentences = [" ".join(s) for s in batch]

            if self.model_args.do_prompt_enhancement and self.model_args.prompt_template:
                # 做提示增强，准备prompt
                template = self.model_args.eval_template if self.model_args.eval_template else self.model_args.prompt_template
                template = template.replace("[MASK]", tokenizer.mask_token)

                for i, s in enumerate(sentences):
                    if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                    sentences[i] = template.replace("{sentence}", s).strip()

            # Tokenization
            if max_length is not None:
                batch = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                    max_length=max_length,
                    truncation=True,
                )
            else:
                batch = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                )

            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(self.device)

            # Get raw embeddings
            with torch.no_grad():

                if self.local_model:
                    outputs = model(**batch, output_hidden_states=True, return_dict=True, **args)
                else:
                    outputs = model(**batch, output_hidden_states=True, return_dict=True)

                last_hidden = outputs.last_hidden_state
                hidden_states = outputs.hidden_states
                pooler_output = outputs.pooler_output

            # Apply different poolers
            if self.model_args.do_prompt_enhancement:
                pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]
                # 如果有应用模板去燥再加
                return pooler_output.view(batch['input_ids'].shape[0], -1).cpu()
            elif pooler == "cls":
                # There is a linear+activation layer after CLS representation
                return pooler_output.cpu()
            elif pooler == "cls_before_pooler":
                return last_hidden[:, 0].cpu()
            elif pooler == "avg":
                return (
                    (last_hidden * batch["attention_mask"].unsqueeze(-1)).sum(1)
                    / batch["attention_mask"].sum(-1).unsqueeze(-1)
                ).cpu()
            elif pooler == "avg_first_last":
                first_hidden = hidden_states[1]
                last_hidden = hidden_states[-1]
                pooled_result = (
                    (first_hidden + last_hidden)
                    / 2.0
                    * batch["attention_mask"].unsqueeze(-1)
                ).sum(1) / batch["attention_mask"].sum(-1).unsqueeze(-1)
                return pooled_result.cpu()
            elif pooler == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = (
                    (last_hidden + second_last_hidden)
                    / 2.0
                    * batch["attention_mask"].unsqueeze(-1)
                ).sum(1) / batch["attention_mask"].sum(-1).unsqueeze(-1)
                return pooled_result.cpu()
            else:
                raise NotImplementedError

        results = {}

        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result

        # Print evaluation results
        if self.mode == "dev":
            logging.info("------ %s ------" % (self.mode))

            task_names = []
            scores = []
            for task in ["STSBenchmark", "SICKRelatedness"]:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
                else:
                    scores.append("0.00")
            print_table(task_names, scores)

            task_names = []
            scores = []
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]["devacc"]))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

        elif self.mode == "test" or self.mode == "fasttest":
            logging.info("------ %s ------" % (self.mode))

            task_names = []
            scores = []
            for task in eval_task_list:
                task_names.append(task)
                if task in results:
                    if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                        scores.append(
                            "%.2f" % (results[task]["all"]["spearman"]["all"] * 100)
                        )
                    else:
                        scores.append(
                            "%.2f" % (results[task]["test"]["spearman"].correlation * 100)
                        )
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

            task_names = []
            scores = []
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]["acc"]))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)
        return results


def main():
    # 解析命令行参数
    parser = HfArgumentParser((EvalArguments, ModelArguments))
    eval_args, model_args = parser.parse_args_into_dataclasses()
    
    eval_util = EvaluationUtil(**eval_args.__dict__, model_args=model_args)

    result = eval_util.eval()
    print(result)


if __name__ == "__main__":
    main()