# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
import sys
import json
import pandas as pd
import argparse
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_en.gym_gen_prompt import *
from utils.utils import load_jsonl_or_parquet
from utils.parquet_utils import list_of_dicts_to_parquet
from utils.online_inference import online_inference

def main(task_file, output_file, cutoff = 1000):
    # 1. Read the parquet file with pandas
    df = load_jsonl_or_parquet(task_file)

    code_gym_task_list = []

    # 2. Iterate over each row of the DataFrame
    for data in tqdm(df):
        if cutoff > 0 and len(code_gym_task_list) >= cutoff:
            break

        code_id = data["question_id"]
        code_task = data["question"]
        code_solution = data["solution"]
        whole_gym_generation_prompt = GYM_GEN_PROMPT.format(
            task_description=TASK_DESCRIPTION,
            example_problem_description=EXAMPLE_CODE_PROBLEM_DESCRIPTION,
            example_solution_code=EXAMPLE_CODE_SOLUTION,
            example_gym_task=EXAMPLE_GYM_TASK,
            example_gym_env=EXAMPLE_GYM_ENV,
            problem_description=code_task,
            solution_code=code_solution,
        )

        code_gym_task = {
            "code_id": code_id,
            "code_task": code_task,
            "code_solution": code_solution,
            "gym_generation_prompt": whole_gym_generation_prompt,
        }

        code_gym_task_list.append(code_gym_task)
        if len(code_gym_task_list) == 1:
            print(code_gym_task_list[0]["gym_generation_prompt"])

    # 3. Dump to parquet or jsonl
    if output_file.endswith(".parquet"):
        list_of_dicts_to_parquet(code_gym_task_list, output_file)
    elif output_file.endswith(".jsonl"):
        with open(output_file, "w") as f:
            for item in code_gym_task_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported data format: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/raw_problems.jsonl", help="The input task file")
    parser.add_argument("--output_file", type=str, default="example/codegym_generation_prompt.jsonl", help="The output file")
    parser.add_argument("--cutoff", type=int, default=10, help="The number of tasks to generate")
    parser.add_argument("--online-inference", action="store_true", default=False, help="Whether to use online inference")
    args = parser.parse_args()
    if args.cutoff > 0 and str(args.cutoff) not in args.output_file:
        args.output_file = args.output_file.replace(".parquet", f"_{args.cutoff}.parquet").replace(".jsonl", f"_{args.cutoff}.jsonl")
    main(task_file = args.input_file, output_file = args.output_file, cutoff = args.cutoff)
    if args.online_inference:
        online_inference(args.output_file, input_key="gym_generation_prompt", output_key="output")