import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Union

import numpy as np
import tqdm
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl
from human_eval.evaluation import estimate_pass_at_k
from human_eval.execution import check_correctness


def maybe_stream_jsonl(filename_or_list: Union[str, List[Dict]]) -> Iterable[Dict]:
    if isinstance(filename_or_list, str):
        return stream_jsonl(filename_or_list)
    else:
        for item in filename_or_list:
            yield item


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(maybe_stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in maybe_stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    return pass_at_k, list(combine_results())


def maybe_increase_indent(completion):
    lines = completion.splitlines()
    for i in range(len(lines)):
        if (
            lines[i].startswith("def")
            or lines[i].startswith("class")
            or lines[i].startswith("import")
            or lines[i].startswith("print")
        ):
            lines = lines[:i]
            break
    need_indent = False
    for i in range(1, len(lines)):
        if not lines[i].startswith("    "):
            need_indent = True
    if need_indent:
        for i in range(len(lines)):
            lines[i] = "    " + lines[i]
    else:
        if not lines[0].startswith("    "):
            lines[0] = "    " + lines[0]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    args = parser.parse_args()

    with open(args.out_path, "r", encoding="utf-8") as f:
        hyp_top1 = json.load(f)
    for i in range(len(hyp_top1)):
        hyp_top1[i]['completion'] = maybe_increase_indent(hyp_top1[i]['completion'][len('<sen001>'):])
        hyp_top1[i]['task_id'] = f"HumanEval/{i}"
    result, full_result = evaluate_functional_correctness(
        hyp_top1,
        k=[1]
    )
    for i in range(len(hyp_top1)):
        if i < 10:
            print("============================================================")
            print(hyp_top1[i]['task_id'])
            print("------------------------------------------------------------")
            print(hyp_top1[i]['completion'])
            print("------------------------------------------------------------")
            print(full_result[i])
            print("============================================================")
    print(result)


if __name__ == '__main__':
    main()
