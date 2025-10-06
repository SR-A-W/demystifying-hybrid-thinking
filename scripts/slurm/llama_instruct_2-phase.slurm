#!/bin/bash
#SBATCH --job-name=HT-analysis_llama_instruct_2-phase
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=<your_slurm_partition>
#SBATCH --account=<your_slurm_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate llms

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN=1

echo "GPUS ON NODE: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

CONFIG_P1="examples/train_full/ht_analysis/llama_instruct_phase1.yaml"
cd train/LLaMA-Factory
llamafactory-cli train "${CONFIG_P1}"

CONFIG_P2="examples/train_full/ht_analysis/llama_instruct_phase2.yaml"
cd train/LLaMA-Factory
llamafactory-cli train "${CONFIG_P2}"

echo "Training finished at: $(date)"