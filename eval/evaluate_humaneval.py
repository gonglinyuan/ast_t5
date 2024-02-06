import argparse
import json

from human_eval.evaluation import evaluate_functional_correctness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    args = parser.parse_args()

    with open(args.out_path, "r", encoding="utf-8") as f:
        hyp_top1 = json.load(f)
    for i in range(len(hyp_top1)):
        hyp_top1[i]['completion'] = hyp_top1[i]['completion'][len('<sen001>'):]
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
