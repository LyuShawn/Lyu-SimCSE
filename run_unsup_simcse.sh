#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

# export CUDA_VISIBLE_DEVICES=2


python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 1e-2 \
    --fp16 \
    --seed 579 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/17 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 5e-2 \
    --fp16 \
    --seed 579 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/18 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 1e-1 \
    --fp16 \
    --seed 579 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/19 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 5e-1 \
    --fp16 \
    --seed 579 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/20 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 1 \
    --fp16 \
    --seed 579 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/21 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --test0_weight 1e-4 \
    --fp16 \
    --seed 579 \
    "$@"