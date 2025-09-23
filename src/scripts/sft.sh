#!/bin/sh

# After running this script, you should save the model's weights
# swift export --adapters __PATH_TO_ADAPTER__ --merge_lora true --output_dir __PATH_TO_MODEL__ --use_hf true

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_API_KEY=__YOUR_WANDB_API_KEY__ \
MAX_PIXELS=50176 \
MASTER_PORT=29500 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --lora_dtype bfloat16 \
    --torch_dtype bfloat16 \
    --dataset "./src/data/qwq_sft_data_red_bbox.json" \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --max_length 8192 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 6 \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir output/rebuttal/image_size/448x448/sft \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --report_to wandb \
    --use_hf true \
