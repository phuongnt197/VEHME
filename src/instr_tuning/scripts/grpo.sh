#!/bin/sh


CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_API_KEY=__YOUR_WANDB_API_KEY__ \
MAX_PIXELS=50176 \
NPROC_PER_NODE=4 \
MASTER_PORT=29500 \
swift rlhf \
    --rlhf_type grpo \
    --model "__YOUR_MODEL_PATH__" \
    --external_plugins ./swift_plugin.py \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --lora_dtype bfloat16 \
    --reward_funcs correctness_match_reward localization_reward thought_length_reward cosine_reward \
    --dataset "__PATH_TO_DATA__" \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --num_generations 6 \
    --eval_steps 200 \
    --save_steps 50 \
    --save_total_limit 50 \
    --logging_steps 1 \
    --output_dir output/__EXP_NAME__ \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --temperature 0.8 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.002 \
    --num_iterations 1 \
    --use_hf true \
    --num_infer_workers 4 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_limit_mm_per_prompt '{"image": 2, "video": 0}' \
    --vllm_max_model_len 2048 \
    