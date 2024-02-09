# AST-T5

Welcome to the repository for AST-T5.

Paper: [AST-T5: Structure-Aware Pretraining for Code Generation and Understanding](https://arxiv.org/abs/2401.03003)

Authors: [Linyuan Gong](https://github.com/gonglinyuan), Mostafa Elhoushi, Alvin Cheung

## Pretraining

For guidelines on pretraining the AST-T5 model, please refer to the instructions in [training/README.md](training/README.md).

## Use the AST-T5 Model

The AST-T5 model is readily available on the Huggingface Model Hub ([https://huggingface.co/gonglinyuan/ast_t5_base](https://huggingface.co/gonglinyuan/ast_t5_base)). To use our AST-T5 model in PyTorch (Python 3.8+, PyTorch 1.12+ and transformers 4.36+ are prerequisites), refer to the code snippet below:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)

input_text = r'''def fibonacci(n):
    """return n-th fibonacci number.
    fibonacci[0] = 0
    fibonacci[1] = 1
    """'''
inputs = tokenizer(
    [input_text + "<sen001>"],  # T5-style sentinel token for completion
    max_length=1024,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt"
).input_ids
outputs = model.generate(inputs, max_length=256, do_sample=False)

output_code = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
output_code = output_code[len("<sen001>"):]  # Remove the sentinel token
print(input_text + output_code)
```

Note: The `ast_t5_base` model is not an instruct model. It works best with specific prompts like function signatures or comments, rather than general instructions such as "Please write a code to calculate the n-th fibonacci number".

## Evaluation

To evaluate AST-T5's performance on HumanEval and MBPP benchmarks and reproduce our results, follow the evaluation instructions in [eval/README.md](eval/README.md).
