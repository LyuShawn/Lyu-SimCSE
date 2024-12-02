from transformers import Trainer
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset
from evaluation import EvaluationUtil

class CLTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        params = EvaluationUtil.prepare_params(kfold=5, optim="rmsprop", batch_size=128, tenacity=3, epoch_size=2)

        tasks = EvaluationUtil.dev_sts_task_list
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks += EvaluationUtil.dev_transfer_task_list

        self.model.eval()
        results = EvaluationUtil.dev_eval(
            model = self.model,
            tokenizer = self.tokenizer,
            tasks = tasks,
            params = params,
            model_args = self.model_args,
        )

        results = EvaluationUtil.process_result(results,tasks,mode="dev")
        metrics = {}
        for task in tasks:
            metrics["eval_{}".format(task)] = results[task]

        metrics["eval_avg_sts"] = (results["STSBenchmark"] + results["SICKRelatedness"]) / 2

        if eval_senteval_transfer or self.args.eval_transfer:
            metrics["eval_avg_transfer"] = sum([results[task] for task in EvaluationUtil.dev_transfer_task_list]) / len(EvaluationUtil.dev_transfer_task_list)

        self.log(metrics)

        return metrics