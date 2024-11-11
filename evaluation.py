import sys
import os
import logging
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer,HfArgumentParser,set_seed
import json
from arguments import ModelArguments,EvalArguments
from datetime import datetime
import random
import wandb

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

def setup_logger(path, epoch):
    """logger"""
    # setup log
    log_path = os.path.join(path, "eval_logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"{epoch}.log")
    # 设置 log
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file, format="%(asctime)s : %(message)s", level=logging.DEBUG
    )


class EvaluationUtil:
    def __init__(self, path, args, times=1, task_set="sts", mode="test", pooler=""):
        """数据准备"""
        self.path = path
        self.args = args
        self.times = times
        self.task_set = task_set
        self.mode = mode

        if not pooler:
            self.pooler = args.pooler_type
        else:
            self.pooler = pooler
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_model = os.path.exists(self.path)
        if not self.local_model:
            # 如果是hf上的模型，只eval一次，不保存log
            self.times = 1
            logger.info(f"load model from huggingface, path:{self.path}")

        task_scores = {task: [] for task in eval_task_list}
        task_scores["avg"] = []
        self.task_scores = task_scores

    def eval(self):
        """评估入口"""
        logger.info(
            f"start evaluation {self.path},with times={self.times},pooler={self.pooler},task_set={self.task_set},mode={self.mode}"
        )

        # 加载模型
        self.model = AutoModel.from_pretrained(self.path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

        for index in range(self.times):
            # 重置随机种子
            seed = self.reset_seed()

            if self.local_model:
                setup_logger(self.path, index)
            result = self.eval_core(
                epoch=index
            )
            # 处理结果，把结果保存到task_scores中
            self.process_result(result, seed)

        if self.local_model:
            score_file_path = os.path.join(self.path, "avg_scores.json")
            with open(score_file_path, "w") as f:
                json.dump(self.scores, f, indent=4, sort_keys=True)
            wandb.log({"score_file": wandb.save(score_file_path)})
        return self.scores

    @property
    def avg_scores(self):
        # 计算平均值
        avg_scores = {task: [] for task in eval_task_list}
        avg_scores["avg"] = []

        task_list = eval_task_list
        task_list.append("avg")

        for task in task_list:
            avg_scores[task] = sum(self.task_scores[task]) / len(self.task_scores[task])
        return avg_scores

    @property
    def scores(self):
        return {"eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"avg_scores": self.avg_scores, "task_scores": self.task_scores}

    def process_result(self, result, seed=None):
        sum = 0
        for task in eval_task_list:
            if task in result:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    self.task_scores[task].append(result[task]["all"]["spearman"]["all"])
                    sum += result[task]["all"]["spearman"]["all"]
                else:
                    self.task_scores[task].append(result[task]["test"]["spearman"].correlation)
                    sum += result[task]["test"]["spearman"].correlation
        self.task_scores["avg"].append(sum / 7)
        if "seed" not in self.task_scores:
            self.task_scores["seed"] = []
        self.task_scores["seed"].append(seed)

    def eval_core(
        self,
        epoch,
    ):
        """评估核心"""

        pooler = self.pooler

        if self.args.do_prompt_denoising:
            noise, template_len = self.get_delta(self.model, self.args.prompt_template, self.tokenizer, self.device, self.args)

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

            if self.args.do_prompt_enhancement and self.args.prompt_template:
                # 做提示增强，准备prompt
                template = self.args.eval_template if self.args.eval_template else self.args.prompt_template
                template = template.replace("[MASK]", self.tokenizer.mask_token)

                for i, s in enumerate(sentences):
                    if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                    sentences[i] = template.replace("{sentence}", s).strip()

            # Tokenization
            if max_length is not None:
                batch = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                    max_length=max_length,
                    truncation=True,
                )
            else:
                batch = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                )

            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(self.device)

            # Get raw embeddings
            with torch.no_grad():

                outputs = self.model(**batch, output_hidden_states=True, return_dict=True)

                try:
                    pooler_output = outputs.pooler_output
                except AttributeError:
                    pooler_output = outputs['last_hidden_state'][:, 0, :]

                if self.args.do_prompt_enhancement:
                    last_hidden = outputs.last_hidden_state
                    pooler_output = last_hidden[batch['input_ids'] == self.tokenizer.mask_token_id]

                    if self.args.do_prompt_denoising:
                            blen = batch['attention_mask'].sum(-1) - template_len
                            pooler_output -= noise[blen] * self.args.prompt_denoising_weight
                else:
                    last_hidden = outputs.last_hidden_state
                    hidden_states = outputs.hidden_states

            # Apply different poolers
            if self.args.do_prompt_enhancement:
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

    def get_delta(self, model, template, tokenizer, device, args):
        model.eval()

        template = template.replace('*mask*', tokenizer.mask_token)\
                        .replace('*sep+*', '')\
                        .replace('*cls*', '').replace('*sent_0*', ' ')
        # strip for roberta tokenizer
        bs_length = len(tokenizer.encode(template.split(' ')[0].replace('_', ' ').strip())) - 2 + 1
        # replace for roberta tokenizer
        batch = tokenizer([template.replace('_', ' ').strip().replace('   ', ' ')], return_tensors='pt')
        batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).to(device).unsqueeze(0)
        for k in batch:
            batch[k] = batch[k].repeat(256, 1).to(device)
        batch['position_ids'][:, bs_length:] += torch.arange(256).to(device).unsqueeze(-1)
        m_mask = batch['input_ids'] == tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(**batch,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            delta = last_hidden[m_mask]
        delta.requires_grad = False
        #import pdb;pdb.set_trace()
        template_len = batch['input_ids'].shape[1]
        return delta, template_len

    def reset_seed(self, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
        set_seed(seed)
        return seed


def main():
    # 解析命令行参数
    parser = HfArgumentParser((EvalArguments, ModelArguments))
    eval_args, model_args = parser.parse_args_into_dataclasses()

    eval_util = EvaluationUtil(path = eval_args.path, 
                            args = model_args, 
                            times=eval_args.times,
                            task_set=eval_args.task_set, 
                            mode = eval_args.mode, 
                            pooler=eval_args.pooler)

    eval_util.eval()


if __name__ == "__main__":
    main()