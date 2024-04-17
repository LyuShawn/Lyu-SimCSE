#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

# export CUDA_VISIBLE_DEVICES=2


python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/23 \
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
    --test0_weight 5e-3 \
    --fp16 \
    --seed 3407 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/24 \
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
    --seed 3407 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/25 \
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
    --seed 3407 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/26 \
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
    --test0_weight 1e-3 \
    --fp16 \
    --seed 10221 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/27 \
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
    --test0_weight 5e-3 \
    --fp16 \
    --seed 10221 \
    "$@"

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/28 \
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
    --seed 10221 \
    "$@"


python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file 'data/simcse_(origin_origin_person).csv' \
    --output_dir result/29 \
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
    --seed 10221 \
    "$@"

