import argparse
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Union
from warnings import warn

import numpy as np
from evalplus.data import (
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.eval import PASS, estimate_pass_at_k
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth
from human_eval.data import stream_jsonl
from tqdm import tqdm


def maybe_stream_jsonl(filename_or_list: Union[str, List[Dict]]) -> Iterable[Dict]:
    if isinstance(filename_or_list, str):
        return stream_jsonl(filename_or_list)
    else:
        for item in filename_or_list:
            yield item


def evaluate_functional_correctness(
    sample_file,
    n_workers: int = 4
):
    problems = get_mbpp_plus()
    dataset_hash = get_mbpp_plus_hash()
    expected_output = get_groundtruth(
        problems,
        dataset_hash,
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": {},
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)  # task_id ->
        remainings = set()

        print("Reading samples...")
        for sample in tqdm(maybe_stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            solution = (
                sample["solution"]
                if "solution" in sample
                else sample["humaneval_style_prompt"] + sample["completion"] + "\n"
            )
            remainings.add(sample["_identifier"])
            args = (
                "mbpp",
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                False,
                True,  # fast_check
                sample["_identifier"],
                1.0,
                4.0,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        assert n_samples == len(remainings), "Missing problems in unfinished"
        assert len(completion_id) == len(problems), "Missing problems in samples"

        def stucking_checker():
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                warn("No samples had finished testing in the last 20s")
                warn(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)

    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = []
        for res in task_results:

            def get_failed_tests(stat, details, inputs) -> List[Any]:
                if stat == PASS or not details:
                    return []

                # esle => simply return the only and the last fail test
                return [inputs[len(details)]]

            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(
                base_stat, base_details, problems[task_id]["base_input"]
            )

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            plus_stat, plus_details = res["plus"]
            plus_fail_tests = get_failed_tests(
                plus_stat, plus_details, problems[task_id]["plus_input"]
            )

            base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
            plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

            results["eval"][task_id].append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fail_tests,
                    "plus_fail_tests": plus_fail_tests,
                }
            )

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        new_correct.append(
            sum(
                [
                    res[i]["base_status"] == res[i]["plus_status"] == PASS
                    for i in range(len(res))
                ]
            )
        )
    base_correct = np.array(base_correct)

    metrics = {}
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    metrics["base"] = pass_at_k

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
        for k in [1, 10, 100]
        if (total >= k).all()
    }
    metrics["plus"] = pass_at_k
    return metrics, results


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
        if lines and not lines[0].startswith("    "):
            lines[0] = "    " + lines[0]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    args = parser.parse_args()

    with open(args.out_path, "r", encoding="utf-8") as f:
        hyp_top1 = json.load(f)
    for i in range(len(hyp_top1)):
        hyp_top1[i]['completion'] = hyp_top1[i]['completion'][len('<sen001>'):]
        hyp_top1[i]['task_id'] = "Mbpp/" + str(hyp_top1[i]["task_id"])
        hyp_top1[i]["_identifier"] = hyp_top1[i]["task_id"] + f" (line {i + 1} in {args.out_path})"
    result, full_result = evaluate_functional_correctness(hyp_top1)
    for i in range(len(hyp_top1)):
        if i < 10:
            print("============================================================")
            print(hyp_top1[i]['task_id'])
            print("------------------------------------------------------------")
            print(hyp_top1[i]['completion'])
            print("------------------------------------------------------------")
            print("base:", full_result['eval'][hyp_top1[i]['task_id']][0]['base_status'])
            print("plus:", full_result['eval'][hyp_top1[i]['task_id']][0]['plus_status'])
            print("============================================================")
    print(result)


if __name__ == '__main__':
    main()
