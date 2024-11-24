from dataclasses import dataclass, field
from typing import Optional, Literal

from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    TrainingArguments,
)

import logging

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=True,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )

    batch_inner_shuffle: bool = field(
        default=False,
        metadata={
            "help": "Whether to shuffle inner batch."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    set_seed_before_shuffle: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to set seed before shuffling the data."
        }
    )

    shuffle_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to shuffle the data."
        }
    )

    sample_data: Optional[int] = field(
        default=None,
        metadata={
            "help": "Sample data."
        }
    )

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default='data/wiki1m_for_simcse.txt', 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    
    # 为一些参数设置适配SimCSE的默认值
    # 训练轮次
    num_train_epochs: float = field(
        default=1.0, 
        metadata={"help": "Total number of training epochs to perform."}
    )
    
    # batch size
    per_device_train_batch_size: int = field(
        default=64, 
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    # 学习率
    learning_rate: float = field(
        default=3e-5, 
        metadata={"help": "The initial learning rate for Adam."}
    )
    
    # 评估策略
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    
    # 评率的最好策略方式
    metric_for_best_model: Optional[str] = field(
        default="eval_STSBenchmark", 
        metadata={"help": "The metric to use to compare two different models."}
    )
    
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    
    eval_steps: int = field(
        default=125, 
        metadata={"help": "Run an evaluation every X steps."}
    )
    
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA Apex) instead of 32-bit"},
    )

    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints."}
    )

    report_to: Optional[str] = field(
        default='all',
        metadata={"help": "The list of integrations to report the results and logs to."}
    )

    logging_steps: int = field(
        default=50,
        metadata={"help": "Log every X updates steps."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode."},
    )

    notes: str = field(
        default="",
        metadata={"help": "Notes."},
    )

    gpu_no: str = field(
        default="0",
        metadata={"help": "GPU number."},
    )

    # 设置torch
    def __post_init__(self):

        if self.gpu_no is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_no

        # 如果没有设置run_name，给wandb设置一个默认的run_name
        if not self.run_name:
            self.run_name = self.output_dir.split("/")[-1]

        super().__post_init__()




@dataclass
class EvalArguments:

    path: str = field(
        metadata={"help": "The path to the model checkpoint."},
    )

    pooler : str = field(
        default="cls",
        metadata={"help": "The pooler type.",
                "choices": ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]},
    )

    mode: str = field(
        default="test",
        metadata={"help": "The evaluation mode.",
                "choices": ["dev", "test","fasttest"]},
    )

    task_set: str = field(
        default="sts",
        metadata={"help": "The task set.",
                "choices": ["sts", "transfer, full","na"]},
    )

    times: int = field(
        default=3,
        metadata={"help": "The number of times to evaluate the model."},
    )

    print_table: bool = field(
        default=False,
        metadata={"help": "Print table."},
    )
