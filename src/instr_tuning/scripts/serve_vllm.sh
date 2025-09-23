#! /bin/sh

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL="../output/path/to/model"

# VLLM_ATTENTION_BACKEND=XFORMERS 
# MAX_NUM=1 \
MAX_PIXELS=50176 \
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
    --host "10.20.22.69" \
    --port 8014 \
    --api-key vehm \
    --max-model-len 8192 \
    --limit-mm-per-prompt "image=2,video=0" \
    --trust-remote-code \
    # --enable-lora \
    # --lora-modules blue-1000=../output/path/to/checkpoint \
