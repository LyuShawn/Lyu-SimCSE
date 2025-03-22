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
from tqdm import tqdm
from mteb.encoder_interface import PromptType
from typing import Optional
import mteb
from mteb.task_selection import results_to_dataframe
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

class CustomMtebModel:
    def __init__(self, model_name, pooler='cls',model=None,tokenizer=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        load_args = {}
        if pooler in ['avg_top2', 'avg_first_last']:
            load_args = {'output_hidden_states': True}

        if "msimcse" in model_name:
            # msimcse没有给词表，与xlmr共用
            self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        else:
            self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)

        self.model = model if model else AutoModel.from_pretrained(model_name, **load_args)
        self.model = self.model.to(self.device)

        self.pooler = Pooler(pooler)
        self.model_name = model_name
        self.model_card_data = {
            "model_name": model_name,
            "pooling": pooler,
            "framework": "PyTorch",
        }

    def encode(
            self,
            sentences: list[str],
            task_name: str,
            prompt_type: Optional[PromptType] = None,
            batch_size: int = 64,
            **kwargs,
        ) -> np.ndarray:
            """Encodes the given sentences using the encoder.

            Args:
                sentences: The sentences to encode.
                task_name: The name of the task.
                prompt_type: The prompt type to use.
                **kwargs: Additional arguments to pass to the encoder.

            Returns:
                The encoded sentences.
            """

            # 实现encode方法
            total = len(sentences)
            for i in tqdm(range(0, total, batch_size), desc="Encoding"):
                batch = sentences[i:i+batch_size]
                inputs = self.tokenizer(batch, padding="longest", truncation=True, return_tensors='pt',max_length=512).to(self.device)
                with torch.no_grad():
                    if self.model_name =='sent_emb':
                        outputs = self.model(**inputs,sent_emb=True)
                    else:
                        outputs =self.model(**inputs)
                    pooler_output = self.pooler(**inputs, outputs=outputs)
                if i == 0:
                    all_embeddings = pooler_output
                else:
                    all_embeddings = torch.cat((all_embeddings, pooler_output), dim=0)
            return all_embeddings.cpu().numpy()


lang_list = 'ar he vi id jv tl eu ml ta te af nl de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw zh kk tr et fi hu az lt pl uk ro'.split()
lang3_dict = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',
    'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml', 'tam':'ta',
    'tel':'te', 'afr':'af', 'nld':'nl', 'eng':'en', 'deu':'de',
    'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',
    'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',
    'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',
    'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',
    'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa', 'aze': 'az',
    'lit': 'lt','pol': 'pl', 'ukr': 'uk', 'ron': 'ro'}
lang_list14 = ['ara', 'bul', 'cmn', 'deu', 'ell', 'fra', 'hin', 'rus', 'spa', 'swh', 'tha', 'tur', 'urd', 'vie']
lang_list36 = lang_list14 + ['afr', 'ben', 'est', 'eus', 'fin', 'heb', 'hun', 'ind', 'ita', 'jav', 'jpn', 'kat', 'kaz', 'kor', 'mal', 'mar', 'nld', 'pes', 'por', 'tam', 'tel', 'tgl']

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

    def __init__(self, path, model_args,bs=128, task_set="sts", mode="test",metric="spearman",mteb_task_set=None, dataset=None, dataset_name=None, *args ,**kwargs):
        """数据准备"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.model_args = model_args
        self.task_set = task_set
        self.metric = metric
        self.bs = bs

        self.print_table_switch = False if not kwargs.get("print_table", None) else kwargs.get("print_table")

        # Set up the tasks
        if self.task_set == "mteb":
            assert mteb_task_set is not None, "mteb_task_set must be specified"
            self.tasks = mteb_task_set.split(',')
        elif self.task_set == "sts":
            self.tasks = self.sts_task_list
        elif self.task_set == "transfer":
            self.tasks = self.transfer_task_list
        elif self.task_set == "full":
            self.tasks = self.sts_task_list + self.transfer_task_list
        else:
            raise NotImplementedError

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
            + (f",mteb_task_set={self.tasks}" if self.task_set == "mteb" else "")
        )

        eval_result = {
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "eval_scores": {},
            "eval_details": [],
        }        

        for path in self.path:

            if self.task_set == "mteb":
                model = CustomMtebModel(path, pooler=self.pooler_type)
                return self.eval_by_mteb(task_name=self.tasks, model=model, bs=self.bs,mode=self.mode)

            else:
                model = AutoModel.from_pretrained(path).to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(path)

                if self.dataset is not None:
                    # 指定数据集
                    result = self.eval_by_dataset(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=self.dataset,
                        dataset_name=self.dataset_name,
                        mode = self.mode,
                        metric=self.metric,
                    )
                    result['avg'] = result[self.metric]
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
                        use_pooler_output = use_pooler_output).cpu()

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
    def eval_by_dataset_core(cls, model, tokenizer,dataset, sent1_name, sent2_name, label_name, bs=64, metric="spearman"):
        """评估核心，与senteval不同的是，这里是自己控制数据集
        评估由自己写
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pooler = Pooler("cls").to(device)

        model.to(device)

        def sent_tokenize(examples):
            total = len(examples[sent1_name])
            sentences = examples[sent1_name] + examples[sent2_name]
            sent_features = tokenizer(sentences, return_tensors="pt", padding='longest', truncation=True, max_length=512)
            features = {}
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            return features

        dataset_tokenize = dataset.map(sent_tokenize, 
                                    batched=True,
                                    load_from_cache_file=True)

        cos_sim_list = []
        # 按照batch_size处理数据
        for batch in tqdm(dataset_tokenize.batch(bs), desc="evaluating"):
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

        if metric == "spearman":
            # 计算spearman相关系数和pearson相关系数
            label_list = np.array(dataset[label_name])
            cos_sim_list = np.array(cos_sim_list)
            spearman_corr, _ = spearmanr(cos_sim_list, label_list)
            pearson_corr, _ = pearsonr(cos_sim_list, label_list)
            return {"spearman": spearman_corr, "pearson": pearson_corr}

        elif metric == "accuracy":
            cos_sim_list = np.array(cos_sim_list)
            cos_sim_list = cos_sim_list > 0.5

            label_list = np.array(dataset[label_name])
            label_list = np.vectorize(lambda x: x == 'true')(label_list)    # 转换为bool

            acc = accuracy_score(label_list, cos_sim_list)
            precision = precision_score(label_list, cos_sim_list)
            recall = recall_score(label_list, cos_sim_list)
            f1 = f1_score(label_list, cos_sim_list)
            return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

        else:
            raise NotImplementedError

    @classmethod
    def eval_by_mteb(cls,task_name,model,bs,mode='test'):

        if task_name[0] == "ChemHotpotQARetrieval":
            tasks = mteb.get_tasks(tasks=["ChemHotpotQARetrieval"],eval_splits=[mode])

            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(model,overwrite_results=True,encode_kwargs={"batch_size": 128})
            result_list = [result.to_dict() for result in results]
            return result_list[0]['scores'][mode]

        elif task_name[0] == "BUCC.v2":
            tasks = mteb.get_tasks(tasks=["BUCC.v2"],eval_splits=[mode])
            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(model,overwrite_results=True,encode_kwargs={"batch_size": 128})
            result_list = [result.to_dict() for result in results]
            return result_list[0]['scores'][mode]

        elif task_name[0].startswith("Tatoeba"):
            lang_list = None
            if task_name[0] == "Tatoeba.14":
                lang_list = lang_list14
                tasks = mteb.get_tasks(tasks=["Tatoeba"],eval_splits=[mode],languages=lang_list14)
            elif task_name[0] == "Tatoeba.36":
                lang_list = lang_list36
                tasks = mteb.get_tasks(tasks=["Tatoeba"],eval_splits=[mode],languages=lang_list36)
            else:
                tasks =  mteb.get_tasks(tasks=["Tatoeba"],eval_splits=[mode])
            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(model,overwrite_results=True,encode_kwargs={"batch_size": bs})
            result_list = [result.to_dict() for result in results]
            result = result_list[0]['scores'][mode]
            if lang_list:
                # 过滤
                result = [r for r in result if r['hf_subset'].split('-')[0] in lang_list]
            avg = sum(r['main_score'] for r in result)/len(result)
            return {'avg': avg, 'result':result}

        else:
            tasks = mteb.get_tasks(tasks=task_name)
            evaluation = mteb.MTEB(tasks=tasks)
            results = evaluation.run(model,overwrite_results=True,encode_kwargs={"batch_size": bs})
            result_list = [result.to_dict() for result in results]
            return result_list

    @classmethod
    def eval_by_dataset(cls, model,tokenizer,dataset, dataset_name, mode, bs=256,metric="spearman", use_mteb=False):
        """自己控制数据集"""

        if use_mteb:
            task_name = dataset_name.split(',')
            model = CustomMtebModel(model_name="tmp", model=model,tokenizer=tokenizer)
            return cls.eval_by_mteb(task_name=task_name,model=model,bs=bs,mode=mode)

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

        elif "Biosses" in dataset_name:
            # Biosses数据集
            if mode == "test":
                dataset = dataset["test"]
            elif mode == "dev":
                dataset = dataset["validation"]
            else:
                raise NotImplementedError
            sent1_name = "sentence1"
            sent2_name = "sentence2"
            label_name = "score"

        elif "mediqa" in dataset_name:
            # mediqa数据集
            if mode == "test":
                dataset = dataset["test"]
            elif mode == "dev":
                dataset = dataset["validation"]
            else:
                raise NotImplementedError
            sent1_name = "text_1"
            sent2_name = "text_2"
            label_name = "label"

        elif "FinSTS" in dataset_name:
            if mode == "test":
                dataset = dataset["test"]
            elif mode == "dev":
                dataset = dataset["validation"]
            else:
                raise NotImplementedError
            sent1_name = "sentence1"
            sent2_name = "sentence2"
            label_name = "score"

        else:
            raise NotImplementedError

        # 按照batch_size处理数据
        return cls.eval_by_dataset_core(model,tokenizer , dataset, sent1_name, sent2_name, label_name, bs=bs, metric=metric)

        

def main():
    # 解析命令行参数
    parser = HfArgumentParser((EvalArguments, ModelArguments))
    eval_args, model_args = parser.parse_args_into_dataclasses()
    
    eval_util = EvaluationUtil(**eval_args.__dict__, model_args=model_args)

    result = eval_util.eval()
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    output_file = 'tmp_result.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Result has been saved to {output_file}")
    print(f"avg:{result['avg']}")


if __name__ == "__main__":
    main()