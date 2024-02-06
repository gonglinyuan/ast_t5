import argparse
import functools
import json

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


def process_generation_example(tokenizer, max_len, examples):
    input_texts = [txt.strip() + "\n    <sen001>" for txt in examples["prompt"]]
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
    parser.add_argument("--max_src_len", type=int)
    parser.add_argument("--max_tgt_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    ds = load_dataset("openai_humaneval")
    raw_dataset = datasets.Dataset.from_list(list(ds['test']))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.eval()
    model.half()
    model.cuda()

    outputs_top1 = []
    for batch in eval_dataloader:
        batch.to(model.device)
        with torch.no_grad():
            gen_out = model.generate(
                **batch,
                num_beams=4,
                length_penalty=0.5,
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
