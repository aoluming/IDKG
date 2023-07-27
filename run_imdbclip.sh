#!/usr/bin/env bash

DATASET_NAME="clipweight"
VILT_NAME="dandelin/vilt-b32-mlm"
CUDA_VISIBLE_DEVICES=0 python -u run_clipimdb.py \
        --dataset_name=${DATASET_NAME} \
        --model_name=${VILT_NAME} \
        --num_epochs=30 \
        --batch_size=64 \
        --lr=0.001 \
        --warmup_ratio=0.06 \
        --modelclass='clipweight' \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --vis_dim=512 \
        --text_dim=512 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path='ckpt/clipkglabelimdb/'