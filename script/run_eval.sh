export CUDA_VISIBLE_DEVICES=2

python evaluation.py \
    --model_name_or_path result/9 \
    --pooler cls \
    --task_set sts \
    --mode test
