#!/bin/bash

script_name=$(basename "$0" .sh)
echo "Starting job: $script_name"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate skythought

export TOKENIZERS_PARALLELISM=false
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

cd <your_eval_results_path>

echo -e "\033[38;2;255;165;0mMATH500 evaluation started\033[0m"
skythought evaluate \
    --model Qwen/Qwen2.5-7B-Instruct \
    --task math500_baseline\
    --backend vllm \
    --backend-args "tensor_parallel_size=1,dtype=float16,gpu_memory_utilization=0.85" \
    --sampling-params "temperature=0.6,top_p=0.95,max_tokens=16384" \
    --n 10 \
    --result-dir ./results/baseline/qwen/

echo -e "\033[38;2;135;206;235mAIME24 evaluation started\033[0m"
skythought evaluate \
    --model Qwen/Qwen2.5-7B-Instruct \
    --task aime24\
    --backend vllm \
    --backend-args "tensor_parallel_size=1,dtype=float16,gpu_memory_utilization=0.85" \
    --sampling-params "temperature=0.6,top_p=0.95,max_tokens=16384" \
    --n 10 \
    --result-dir ./results/baseline/qwen/

echo -e "\033[38;2;135;206;235mGPQA_DIAMOND evaluation started\033[0m"
skythought evaluate \
    --model Qwen/Qwen2.5-7B-Instruct \
    --task gpqa_diamond\
    --backend vllm \
    --backend-args "tensor_parallel_size=1,dtype=float16,gpu_memory_utilization=0.85" \
    --sampling-params "temperature=0.6,top_p=0.95,max_tokens=16384" \
    --n 10 \
    --result-dir ./results/baseline/qwen/

echo "Evaluation finished at: $(date)"