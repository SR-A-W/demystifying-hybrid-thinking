# Scripts

This directory contains example scripts for running training and evaluation tasks on different computing environments.

## Script Types

### Shell Scripts (`sh/`)
- **eval_example.sh**: Example script for running evaluation tasks on local machines or servers
- **train_example.sh**: Example script for running training tasks on local machines or servers

### SLURM Scripts (`slurm/`)
- **eval_example.slurm**: Example SLURM script for running evaluation tasks on HPC clusters
- **llama_instruct_2-phase.slurm**: Example SLURM script for running 2-phase training on HPC clusters

## Usage

1. **For local machines or servers**: Use the scripts in the `sh/` directory
2. **For HPC clusters**: Use the scripts in the `slurm/` directory and submit them using `sbatch`

## Note