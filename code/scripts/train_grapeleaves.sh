#!/bin/bash

deepspeed --num_gpus=1 --master_port 28400 train_grapeleaves.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --max_tgt_len 1024 \
    --data_path ./grapeleaves_instruction_data.json \
    --image_root_path ../data/grapeleaves \
    --save_path ./ckpt/train_grapeleaves/ \
    --log_path ./ckpt/train_grapeleaves/log_rest/
