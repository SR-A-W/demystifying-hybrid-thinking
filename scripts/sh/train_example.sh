#!/bin/bash
set -euo pipefail

source ~/.bashrc
conda activate skythought

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export WORLD_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    echo "Distributed training setup: WORLD_SIZE=${WORLD_SIZE}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

cd train/LLaMA-Factory

echo "Starting Phase 1 training at: $(date)"
CONFIG_P1="examples/train_full/ht_analysis/llama_phase1.yaml"
llamafactory-cli train "${CONFIG_P1}"
echo "Phase 1 training completed at: $(date)"

echo "Starting Phase 2 training at: $(date)"
CONFIG_P2="examples/train_full/ht_analysis/llama_phase2.yaml"
llamafactory-cli train "${CONFIG_P2}"
echo "Phase 2 training completed at: $(date)"
echo "All training finished at: $(date)"
