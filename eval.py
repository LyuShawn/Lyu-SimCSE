import sys
import os
import logging
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer
import json

PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval

logger = logging.getLogger(__name__)

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


def do_eval(
    model_name_or_path,
    pooler,
    mode,
    task_set,
    epoch,
    tasks=None,
):

    # load model
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up the tasks
    if task_set == "sts":
        tasks = eval_task_list
    elif task_set == "transfer":
        tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif task_set == "full":
        tasks = eval_task_list
        tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

    # Set params for SentEval
    if mode == "dev" or mode == "fasttest":
        # Fast mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
    elif mode == "test":
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
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if pooler == "cls":
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
    if mode == "dev":
        logging.info("------ %s ------" % (mode))

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

    elif mode == "test" or mode == "fasttest":
        logging.info("------ %s ------" % (mode))

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


def process_result(result, task_scores):
    sum = 0
    for task in eval_task_list:
        if task in result:
            if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                task_scores[task].append(result[task]["all"]["spearman"]["all"])
                sum += result[task]["all"]["spearman"]["all"]
            else:
                task_scores[task].append(result[task]["test"]["spearman"].correlation)
                sum += result[task]["test"]["spearman"].correlation
    task_scores["avg"].append(sum / 7)


def calculate_average(task_scores):
    avg_scores = {
        "STS12": [],
        "STS13": [],
        "STS14": [],
        "STS15": [],
        "STS16": [],
        "STSBenchmark": [],
        "SICKRelatedness": [],
        "avg": [],
    }
    for task in [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICKRelatedness",
        "avg",
    ]:
        avg_scores[task] = sum(task_scores[task]) / len(task_scores[task])
    return avg_scores

def setup_logger(model_path, epoch):
    """logger"""
     # setup log
    log_path = os.path.join(model_path, "eval_logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"{epoch}.log")
    # 设置 log
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file, format="%(asctime)s : %(message)s", level=logging.DEBUG
    )

def eval(path, times=3, pooler="cls", task_set="sts", mode="test"):
    """推断模型函数入口"""
    logger.info(
        f"start evaluation {path},with times={times},pooler={pooler},task_set={task_set},mode={mode}"
    )
    task_scores = {
        "STS12": [],
        "STS13": [],
        "STS14": [],
        "STS15": [],
        "STS16": [],
        "STSBenchmark": [],
        "SICKRelatedness": [],
        "avg": [],
    }

    for index in range(times):
        setup_logger(path, index)
        result = do_eval(
            model_name_or_path=path,
            pooler=pooler,
            mode=mode,
            task_set=task_set,
            epoch=index,
        )
        process_result(result, task_scores)

    avg_scores = calculate_average(task_scores)
    scores = {"avg_scores": avg_scores, "task_scores": task_scores}
    with open(os.path.join(path, "avg_scores.json"), "w") as f:
        json.dump(scores, f, indent=4, sort_keys=True)
    return scores