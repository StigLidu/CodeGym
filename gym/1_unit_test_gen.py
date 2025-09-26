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
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_en.unit_test_gen_prompt import UNIT_TEST_GENERATION_PROMPT
from utils.online_inference import online_inference
from utils.utils import load_jsonl_or_parquet
from utils.parquet_utils import list_of_dicts_to_parquet

def main(data_path, save_path):
    data_list = load_jsonl_or_parquet(data_path)
    unit_test_prompt_list = []
    for data in tqdm(data_list):
        gym_task = data["gym_task"]
        gym_env = data["gym_env"]
        unit_test_prompt = UNIT_TEST_GENERATION_PROMPT.format(
            problem_description=gym_task,
            gym_env=gym_env
        )

        unit_test_prompt_list.append({
            "code_id": data["code_id"],
            "idx": data["idx"],
            "env_name": data["env_name"],
            "gym_task": gym_task,
            "gym_env": gym_env,
            "unit_test_prompt": unit_test_prompt
        })

        # print an example
        if len(unit_test_prompt_list) == 1:
            print(unit_test_prompt_list[0]["unit_test_prompt"])
    
    # save into parquet or jsonl
    if save_path.endswith(".parquet"):
        list_of_dicts_to_parquet(unit_test_prompt_list, save_path)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for item in unit_test_prompt_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported data format: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/codegym_envs.jsonl", help="the input file to read from")
    parser.add_argument("--output_file", type=str, default="example/codegym_unit_test_prompt.jsonl", help="the output file to save to")
    parser.add_argument("--online-inference", action="store_true", default=False, help="whether to use online inference")
    args = parser.parse_args()
    main(data_path=args.input_file, save_path=args.output_file)
    if args.online_inference:
        online_inference(args.output_file, input_key="unit_test_prompt", output_key="output", num = 2, repeat = 1)