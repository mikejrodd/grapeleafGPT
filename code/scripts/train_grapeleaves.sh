#!/bin/bash

# Ensure the script uses train_grapeleaves.py instead of train_mvtec.py
deepspeed --num_gpus=1 --master_port 28400 train_grapeleaves.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --max_tgt_len 1024 \
    --data_path ../data/pandagpt4_visual_instruction_data.json \
    --image_root_path ../data/images/ \
    --save_path ./ckpt/train_grapeleaves/ \
    --log_path ./ckpt/train_grapeleaves/log_rest/