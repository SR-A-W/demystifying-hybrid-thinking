## Training
For model training, we use a fork from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Adding New Tasks
To use/add new training tasks, please follow the steps below:

1. Please add the dataset file under [data](./data) directory, and update [data/dataset_info.json](./data/dataset_info.json) correspondingly.

2. Please add your experiment configuration `.yaml` files under [tasks](./tasks) directory. You can organize them in subdirectories based on your experiment categories (e.g., `factors/`, `local_demo/`). If you need to customize your task, you can construct your own `.yaml` file.

### Task Configuration Categories

The [tasks](./tasks) directory contains the following experiment categories:

- **`local_demo/`**: Local smoke tests that can be run on consumer hardware
  - Designed for quick testing and validation
  - Requires only 16GB GPU memory
  - Perfect for local development and debugging

- **`factors/`**: Experimental configurations for studying different factors
  - **`2-phase/`**: Training configurations for 2-phase/mix ablation study with different dataset sizes
  - **`pairs/`**: Training configurations for pairs/without_pairs ablation studies
  - **`ratio/`**: Training configurations for /think : /no_think sample ratio ablation studies
    - **`2-phase/`**: Two-phase training with different think/no-think ratios
    - **`mix/`**: Mixed training with with different think/no-think ratios

### Running SFT 

#### General Training Instructions
To run training tasks, please follow the steps below:
1. **Install LLaMA-Factory**: If you haven't yet, install LLaMA-Factory following the instruction in LLaMA-Factory's [README.md](https://github.com/SR-A-W/LLaMA-Factory/blob/main/README.md) (by using `pip install -e ".[torch,metrics]" --no-build-isolation`). 
2. To run a training task, use LLaMA-Factory's command lines, for example, `llamafactory-cli train train/tasks/local_demo/qwen0.5_phase1_20k.yaml`

To train 7/8B models, we recommend using 2 H200 GPUs, 24 CPUs, or equivalent.

#### Quick Start (Local Demo)
For a quick smoke test that can run on consumer hardware (16GB GPU):

1. Install LLaMA-Factory (if not already installed):
   See [README.md](https://github.com/SR-A-W/LLaMA-Factory/blob/main/README.md)

2. Navigate to the project root directory:
   ```bash
   cd /path/to/demystifying-hybrid-thinking
   ```

3. Run the local demo training:
   ```bash
   llamafactory-cli train train/tasks/local_demo/qwen0.5_phase1_20k.yaml
   ```
