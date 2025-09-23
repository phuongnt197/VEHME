#!/bin/sh

CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=50176 \
swift deploy \
    --model "__YOUR_MODEL_PATH__" \
    --infer_backend vllm \
    --host "__YOUR_IP_ADDRESS__" \
    --port 8010 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --tensor-parallel-size 4 \
    --limit_mm_per_prompt '{"image": 2, "video": 0}' \
    --served_model_name __WHATEVERYOULIKE__ \
    --use_hf true \
