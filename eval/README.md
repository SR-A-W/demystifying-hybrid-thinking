## Evaluation
For evaluation tasks, we use a fork from [SkyThought](https://github.com/NovaSky-AI/SkyThought/tree/main).

### Adding New Tasks
To add new training tasks, please construct your own `.yaml` file under [skythought/skythought/evals/tasks](./skythought/skythought/evals/tasks). You can also just use our settings under [skythought/skythought/evals/tasks/ht](./skythought/skythought/evals/tasks/ht) directory. 

### Running Evaluation 
To run evaluation tasks, please follow the steps below:
1. If you haven't yet, install skythought following the instruction in skythought's [README.md](./skythought/README.md) (by using `pip install -e .`). 
2. To run a evaluation task, use skythought's command lines, for example, `skythought evaluate --model Qwen/Qwen2.5-7B-Instruct --task math500_baseline --backend vllm --sampling-params "temperature=0.6,top_p=0.95,max_tokens=16384" --n 10 --result-dir ./results/baseline/qwen/`

