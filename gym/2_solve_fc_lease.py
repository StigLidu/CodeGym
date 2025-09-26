# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import json
import os
import sys
import shutil
import concurrent.futures
from tqdm import tqdm
import argparse
import pandas as pd
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from utils.utils import load_jsonl_or_parquet
from utils.parquet_utils import list_of_dicts_to_parquet

def check_env_wrapper(env, check_docstring=True, check_correctness=True):
    return env if utils.check_env(env["gym_env"], check_docstring=check_docstring, check_correctness=check_correctness) else None

def main(source_file, target_file, force_mode=False, num_processes=32):
    # remove target folder if exists
    if os.path.exists(target_file) and force_mode:
        os.remove(target_file)
    else:
        assert not os.path.exists(target_file), f"target file {target_file} already exists"

    env_list = []
    data_list = load_jsonl_or_parquet(source_file)
    total_data_count = sum([1 for _ in data_list])

    for data in tqdm(data_list, total=total_data_count):
        original_solve_fc, o_start_line, o_end_line = utils.pass_fc_from_output(data["gym_env"], "solve", indent_level=4)
        for i, output in enumerate(data["output"]):
            try:
                solve_fc, s_start_line, s_end_line = utils.pass_fc_from_output(output, "solve", indent_level=0)
                solve_fc = utils.add_indent(solve_fc, 4)
            except Exception as e:
                # the solve fc cannot be extracted from the output
                continue

            if solve_fc is None:
                continue

            # replace the original solve fc with the new solve fc
            assert original_solve_fc in data["gym_env"], f"original solve fc {original_solve_fc} not in gym env {data['gym_env']}"
            new_gym_env = data["gym_env"].replace(original_solve_fc, solve_fc)

            # save the new gym env
            env_list.append({key: value for key, value in data.items() if key != "gym_env"})
            env_list[-1]["gym_env"] = new_gym_env
            env_list[-1]["docstring"] = data["docstring_prompt"]
            env_list[-1]["temp_idx"] = len(env_list) - 1

    env_list_passed = []
    passed_code_id_set = set()

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(check_env_wrapper, env_list), total=len(env_list), desc="Checking envs"))

    env_list_passed = [env for env in env_list if results[env["temp_idx"]]]
    passed_code_id_set = list(set([env["code_id"] for env in env_list_passed]))

    # pop the temp_idx
    for env in env_list:
        env.pop("temp_idx")

    print(f"Step 2: {len(env_list_passed)} / {len(env_list)} envs passed compilation test. Finished.")
    print(f"Step 2: {len(passed_code_id_set)} code_id passed compilation test. Finished.")

    # save into parquet or jsonl
    if target_file.endswith(".parquet"):
        list_of_dicts_to_parquet(env_list_passed, target_file)
    elif target_file.endswith(".jsonl"):
        with open(target_file, "w") as f:
            for item in env_list_passed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported data format: {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/codegym_solve_fc_prompt_output.jsonl", help="the input file to read from")
    parser.add_argument("--target_file", type=str, default="example/codegym_solve_fc.jsonl", help="the target file to save the envs")
    parser.add_argument("--force_mode", action="store_true", default=True, help="whether to force overwrite the target folder")
    parser.add_argument("--num_processes", type=int, default=128, help="the number of processes to use")
    args = parser.parse_args()
    main(source_file=args.input_file, target_file=args.target_file, force_mode=args.force_mode, num_processes=args.num_processes)