# AST-T5: Evaluation on HumanEval and MBPP Datasets

Below, you will find instructions on setting up the environment and reproducing the experimental results as presented in our paper.

## Setup the Environment

First, ensure you have Anaconda or Miniconda installed to manage environments. Then, create and activate the `ast_t5` environment by running the following commands:

```bash
conda create -n ast_t5 python=3.8.10
conda install -n ast_t5 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate ast_t5
python -m pip install transformers==4.36.0 "omegaconf<2.1" tiktoken==0.5.2 datasets human_eval pyarrow==8.0.0 astunparse evalplus
```

## Inference and Evaluation

To perform inference and evaluate the model's performance on the HumanEval and MBPP benchmarks, execute the following scripts. These steps will generate output files and report the expected performance metrics.

### HumanEval

```bash
python run_humaneval.py gonglinyuan/ast_t5_base \
--out_path astt5_humaneval_outputs.json

python evaluate_humaneval.py astt5_humaneval_outputs.json 
# Expected output: {'pass@1': 0.1402439024390244}
```

### MBPP

```bash
python run_mbpp.py gonglinyuan/ast_t5_base \
--out_path mbpp_humaneval_outputs.json

python evaluate_mbpp.py mbpp_humaneval_outputs.json 
# Expected output: {'pass@1': 0.2388758782201405}
```

### HumanEvalPlus

```bash
python run_humaneval_plus.py gonglinyuan/ast_t5_base \
--out_path astt5_humaneval_plus_outputs.json

python evaluate_humaneval_plus.py astt5_humaneval_plus_outputs.json 
# Expected output: {'base': {'pass@1': 0.1402439024390244}, 'plus': {'pass@1': 0.12804878048780488}}
```