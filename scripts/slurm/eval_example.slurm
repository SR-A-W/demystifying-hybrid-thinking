#!/bin/bash
#SBATCH --job-name=eval_ht-analysis_llama_no_think-only_array
#SBATCH --partition=<your_slurm_partition>
#SBATCH --account=<your_slurm_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x/%x_%A_%a.out
#SBATCH --error=logs/%x/%x_%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

MODEL_PATH="<your_model_path>"
THINK_RESULT_DIR="./results/ht-analysis/llama/llama_no_think-only/think"
NO_THINK_RESULT_DIR="./results/ht-analysis/llama/llama_no_think-only/no_think"

source ~/.bashrc
conda activate llms

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

echo "JOB_ID: ${SLURM_JOB_ID:-na}  ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-na}"
echo "GPUS ON NODE: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

cd <your_skythought_path>

TASKS=(math500 aime24 mmlu_stem gpqa_diamond)
MODES=(think no_think)
NS=(10 10 1 10)

a=${SLURM_ARRAY_TASK_ID:-0}
task_idx=$(( a / 2 ))
mode_idx=$(( a % 2 ))

TASK="${TASKS[$task_idx]}"
MODE="${MODES[$mode_idx]}"
N="${NS[$task_idx]}"

TASK_FLAG="${TASK}_${MODE}"

if [[ "$MODE" == "think" ]]; then
  RESULT_DIR="$THINK_RESULT_DIR"
else
  RESULT_DIR="$NO_THINK_RESULT_DIR"
fi

mkdir -p "$RESULT_DIR"

echo -e "\033[38;2;255;165;0m[${TASK^^}] evaluation started\033[0m"
echo -e "\033[38;2;135;206;235mEvaluating ${MODE}\033[0m"
echo "Resolved: --task ${TASK_FLAG}, --n ${N}, --result-dir ${RESULT_DIR}"
srun skythought evaluate \
    --model "$MODEL_PATH" \
    --task "$TASK_FLAG" \
    --backend vllm \
    --backend-args "tensor_parallel_size=1,dtype=float16,gpu_memory_utilization=0.85" \
    --sampling-params "temperature=0.6,top_p=0.95,max_tokens=16384" \
    --n "$N" \
    --result-dir "$RESULT_DIR"

echo "Evaluation finished at: $(date)"
