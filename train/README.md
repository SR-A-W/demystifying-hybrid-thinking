## Training
For model training, we use a fork from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Adding New Tasks
To use/add new training tasks, please follow the steps below:

1. Please add the dataset file under [LLaMA-Factory/data](./LLaMA-Factory/data) directory, and update [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json) correspondingly.

2. Please use the sft settings under [LLaMA-Factory/examples/train_full](./LLaMA-Factory/examples/train_full) directory. If you need to customize your task, you can construct your own `.yaml` file.

### Running SFT 
To run training tasks, please follow the steps below:
1. If you haven't yet, install LLaMA-Factory following the instruction in LLaMA-Factory's [README.md](./LLaMA-Factory/README.md) (by using `pip install -e ".[torch,metrics]" --no-build-isolation`). 
2. To run a training task, use LLaMA-Factory's command lines, for example, `llamafactory-cli train examples/train_full/qwen2_full_sft.yaml`

To train 7/8B models, we recommend using 2 H200 GPUs, 24 CPUs, or equivalent.