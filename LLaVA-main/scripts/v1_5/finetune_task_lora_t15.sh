#!/bin/bash
export WANDB_MODE=offline

deepspeed /root/autodl-tmp/LLaVA-main/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /root/autodl-tmp/LLaVA-main/scripts/zero3.json \
    --model_name_or_path /root/autodl-tmp/llava-v1.5-7b \
    --version v1 \
    --data_path /root/autodl-tmp/LLaVA-main/playground/data/t15_train.json \
    --image_folder /root/autodl-tmp/LLaVA-main/playground/data \
    --vision_tower /root/autodl-tmp/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir /root/autodl-tmp/LLaVA-main/checkpoints/llava-v1.5-7b-sentiment-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --max_grad_norm 1.0 \
    --report_to wandb
