import sys
import warnings
from transformers import Trainer
from transformers.utils import logging

from transformers.utils import logging
import torch
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset
from torch.amp import autocast


PATH_TO_DATA = "./SentEval/data"

import SentEval.senteval as senteval

logger = logging.get_logger(__name__)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CLTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [" ".join(s) for s in batch]
            batch = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
            ).to(self.args.device)

            with torch.no_grad():
                with autocast(device_type=self.args.device.type):
                    outputs = self.model(
                        **batch, output_hidden_states=True, return_dict=True, sent_emb=True
                    )
                    pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ["STSBenchmark", "SICKRelatedness"]
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = [
                "STSBenchmark",
                "SICKRelatedness",
                "MR",
                "CR",
                "SUBJ",
                "MPQA",
                "SST2",
                "TREC",
                "MRPC",
            ]
        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results["STSBenchmark"]["dev"]["spearman"][0]
        sickr_spearman = results["SICKRelatedness"]["dev"]["spearman"][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,
        }
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                avg_transfer += results[task]["devacc"]
                metrics["eval_{}".format(task)] = results[task]["devacc"]
            avg_transfer /= 7
            metrics["eval_avg_transfer"] = avg_transfer

        # self.log(metrics)
        logger.info(metrics)
        return metrics
