This repository is the official implementation for our paper, "Demystifying Hybrid Thinking: Cam LLMs Truly Switch Between Think And No-Think?"".

## Repository Structure

- **`train/`**: Training code using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **`eval/`**: Evaluation code using [SkyThought](https://github.com/NovaSky-AI/SkyThought)
- **`utils/`**: Dataset construction, processing, and utility scripts
- **`scripts/`**: Training and evaluation task scripts (`.sh` for shell scripts, `.slurm` for HPC clusters)

## Quick Start

1. **Training**: See [train/README.md](train/README.md) for training instructions
2. **Evaluation**: See [eval/README.md](eval/README.md) for evaluation instructions
3. **Utilities**: See [utils/README.md](utils/README.md) for dataset processing utilities
4. **Scripts**: See [scripts/README.md](scripts/README.md) for example execution scripts
