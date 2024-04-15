import argparse
import ast
import copy
import functools
import json

import astunparse
import datasets
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


def convert_mbpp_example(sample):
    tree = ast.parse(sample["test_list"][0])
    if isinstance(tree.body[0].test, ast.Call):
        if isinstance(tree.body[0].test.func, ast.Attribute):
            func_name = tree.body[0].test.args[0].func
        else:
            func_name = tree.body[0].test.func.id
    elif isinstance(tree.body[0].test, ast.UnaryOp) and isinstance(tree.body[0].test.op, ast.Not):
        func_name = tree.body[0].test.operand.func.id
    else:
        func_name = tree.body[0].test.left.func.id
    tree = ast.parse(sample["code"])
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node = copy.copy(node)
            break
    docstring = sample["prompt"] + "\n"
    for test_case in sample["test_list"]:
        tree = ast.parse(test_case)
        if isinstance(tree.body[0].test, ast.Call):
            docstring += ">>> " + astunparse.unparse(tree.body[0].test)
            docstring += "True\n"
        elif isinstance(tree.body[0].test, ast.UnaryOp) and isinstance(tree.body[0].test.op, ast.Not):
            docstring += ">>> " + astunparse.unparse(tree.body[0].test.operand)
            docstring += "False\n"
        else:
            docstring += ">>> " + astunparse.unparse(tree.body[0].test.left)
            assert len(tree.body[0].test.ops) == 1 and isinstance(tree.body[0].test.ops[0], ast.Eq)
            docstring += astunparse.unparse(tree.body[0].test.comparators[0])
    node.body = [
        ast.Expr(value=ast.Constant(value="MBPP_PLACEHOLDER", kind=None))
    ]
    docstring = '"""\n' + "".join(["    " + line + "\n" for line in docstring.splitlines()]) + '    """'
    result = astunparse.unparse(node).replace("'MBPP_PLACEHOLDER'", docstring).strip()
    result = result + "\n    "
    return result


def process_generation_example(tokenizer, max_len, examples):
    input_texts = [s + "<sen001>" for s in examples["humaneval_style_prompt"]]
    return tokenizer(
        input_texts,
        padding=False,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    ds = datasets.load_dataset("evalplus/mbppplus")
    m = list(ds['prompt']) + list(ds['train']) + list(ds['validation']) + list(ds['test'])
    for mm in m:
        mm['humaneval_style_prompt'] = convert_mbpp_example(mm)
    raw_dataset = datasets.Dataset.from_list(m)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    process_fn = functools.partial(process_generation_example, tokenizer, args.max_src_len)
    eval_dataset = raw_dataset.map(
        process_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        num_proc=4
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding="longest",
        max_length=args.max_src_len,
        pad_to_multiple_of=8,
        label_pad_token_id=tokenizer.pad_token_id,
        return_tensors="pt"
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    model.half()
    model.cuda()

    outputs_top1 = []
    for batch in tqdm(eval_dataloader):
        batch.to(model.device)
        with torch.no_grad():
            gen_out = model.generate(
                **batch,
                num_beams=4,
                num_return_sequences=1,
                max_length=args.max_tgt_len
            )
            assert batch['input_ids'].size(0) == gen_out.size(0)
            for i in range(batch['input_ids'].size(0)):
                hyp = tokenizer.decode(
                    gen_out[i],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                outputs_top1.append({"completion": hyp})

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(outputs_top1, f)


if __name__ == '__main__':
    main()
