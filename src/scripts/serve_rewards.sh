#! /bin/sh

MODEL="Qwen/QwQ-32B-AWQ"

# VLLM_ATTENTION_BACKEND=XFORMERS 
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
    --host "__YOUR_IP_ADDRESS__" \
    --port 8011 \
    --api-key "__YOUR_API_KEY__" \
    --max-model-len 4096 \
    --trust-remote-code \
