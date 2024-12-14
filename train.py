import logging
import os
import argparse
from evaluation import EvaluationUtil

from datasets import load_dataset

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    BertForPreTraining,
)
from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer
from loader.loader import PrepareFeaturesArgs, prepare_features
from loader.collator import OurDataCollatorWithPadding
from arguments import (ModelArguments, DataTrainingArguments, OurTrainingArguments)
import wandb
import torch

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 检查模型输出
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    exp_name = training_args.output_dir.split('/')[-1]

    # wandb
    wandb_project = "Train_Debug" if training_args.debug_mode else "LyuCSE"
    notes = training_args.notes if training_args.notes else None
    wandb.init(project=wandb_project, 
                name=exp_name, save_code=True, notes=notes,)
    logger.info(f"*** Wandb Will Save : {wandb.run.project}/{wandb.run.name} ***")


    # 打印参数
    logger.info("********* Arguments *********")
    logger.info(f"model args: {model_args.__dict__}")
    logger.info(f"data args: {data_args.__dict__}")
    logger.info(f"training args: {training_args.__dict__}")

    # 加载数据集
    logger.info("********* Load Dataset *********")
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/cache/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/cache/")

    if data_args.set_seed_before_shuffle:
        # 在shuffle之前设置随机种子，可以保证每次shuffle的结果一样
        set_seed(training_args.seed)

    # 是否随机打乱数据
    if data_args.shuffle_data:
        # 这里的执行位置会受到到set_seed的影响
        datasets = datasets.shuffle()

    # 是否采样数据
    if data_args.sample_data:
        # 只有train所以只对train进行采样
        datasets["train"] = datasets["train"].select(range(data_args.sample_data))

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载模型参数
    logger.info("********* Load Model Config *********")
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.hidden_dropout_prob=model_args.dropout
    config.attention_probs_dropout_prob=model_args.dropout

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    # 加载模型
    logger.info("********* Load Model *********")
    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # 更新tokenizer的词表大小
    model.resize_token_embeddings(len(tokenizer))

    # 数据tokenize
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # 两个句子的数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # 三个句子的数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # 单个句子的数据集
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError


    model.mask_token_id = tokenizer.mask_token_id
    model.pad_token_id = tokenizer.pad_token_id
    if model_args.do_prompt_enhancement:

        template = model_args.prompt_template.replace('[MASK]', tokenizer.mask_token)
        model_args.template = template

        model_args.prompt_prefix = template.split('{sentence}')[0]    
        model_args.prompt_suffix = template.split('{sentence}')[1]

        eval_template = model_args.eval_template if model_args.eval_template else model_args.prompt_template
        eval_template = eval_template.replace('[MASK]', tokenizer.mask_token)
        model_args.eval_template = eval_template
        model_args.eval_prefix = eval_template.split('{sentence}')[0]
        model_args.eval_suffix = eval_template.split('{sentence}')[1]

        model.eval_prefix_origin_input_ids = tokenizer.encode(model_args.eval_prefix, add_special_tokens=False)
        model.eval_suffix_origin_input_ids = tokenizer.encode(model_args.eval_suffix, add_special_tokens=False)

        # logger
        logger.info(f"*** Prompt Template : {model_args.prompt_template} ***")
        logger.info(f"*** Eval Template : {model_args.eval_template} ***")


    prepare_features_args = PrepareFeaturesArgs(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        sent0_cname=sent0_cname,
        sent1_cname=sent1_cname,
        sent2_cname=sent2_cname
    )

    if training_args.do_train:
        # 准备训练数据
        logger.info("********* Prepare Train Dataset *********")
        train_dataset = datasets["train"].map(
            lambda x: prepare_features(x, prepare_features_args),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file= not data_args.overwrite_cache,
        )

    if data_args.pad_to_max_length:
        # 如果需要将数据填充到最大长度，使用默认的collator
        data_collator = default_data_collator
    else:
        # 否则使用自定义的collator，传入tokenizer
        data_collator = OurDataCollatorWithPadding(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, do_mlm=model_args.do_mlm, model_args=model_args)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(resume_from_checkpoint=model_path)
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:

        logger.info(f"*** Evaluate ***")
        eval_util = EvaluationUtil(path = training_args.output_dir, model_args = model_args, print_table=True)
        results, result_file_path = eval_util.eval()
        wandb.log(results)
        wandb.log({"score_file": wandb.save(result_file_path)})

    logger.info("********* Finish *********")
    wandb.finish()
    return results


if __name__ == "__main__":

    main()