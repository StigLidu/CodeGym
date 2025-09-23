import json
import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
import argparse
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from utils.parquet_utils import list_of_dicts_to_parquet

def check_env_wrapper(env, check_docstring=True):
    return env if utils.check_env(env["gym_env"], check_docstring=check_docstring) else None

def main(input_file, target_file, force_mode=False, cutoff=10000, num_processes=32):
    """
        save format: code_id__idx__env_name.py
        save content:
            - task.txt: the task description
            - env.py: the gym environment code
    """
    # remove target file if exists and force_mode is True
    if os.path.exists(target_file) and force_mode:
        os.remove(target_file)
    else:
        if os.path.exists(target_file):
            raise FileExistsError(f"Target file {target_file} already exists. Please use --force_mode to overwrite it.")
        
    env_list = []
    total_env_count = 0

    with open(input_file, "r") as f:
        for line in tqdm(f):
            if total_env_count >= cutoff:
                break
            data = json.loads(line)
            for idx in range(len(data["output"])):
                total_env_count += 1
                try:
                    assert "<task>" in data["output"][idx] and "</task>" in data["output"][idx] and "<env>" in data["output"][idx] and "</env>" in data["output"][idx]
                    gym_task = data["output"][idx].split("<task>")[-1].split("</task>")[0].strip()
                    gym_env = data["output"][idx].split("<env>")[-1].split("</env>")[0].strip()
                    #class StampCombinationEnv(gymnasium.Env):
                    gym_env_name = gym_env.split("class ")[-1].split("(")[0].strip()
                    assert len(gym_env_name) > 0 and len(gym_env_name) < 100 and gym_env_name.isalpha()
                except Exception as e:
                    continue

                env_list.append({
                    "code_id": data["code_id"],
                    "idx": idx,
                    "env_name": gym_env_name,
                    "gym_task": gym_task,
                    "gym_env": gym_env,
                    "temp_idx": len(env_list)
                })
    
    print(f"Step 1: extract {len(env_list)} / {total_env_count} envs. Finished.")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(check_env_wrapper, env_list), total=len(env_list), desc="Checking envs"))

    env_list_passed = [env for env in env_list if results[env["temp_idx"]]]

    # save one env for each code_id
    passed_code_id_map = dict()
    for env in env_list_passed:
        if env["code_id"] not in passed_code_id_map:
            passed_code_id_map[env["code_id"]] = env
    
    dedup_env_list_passed = list(passed_code_id_map.values())

    # pop the temp_idx
    for env in env_list:
        env.pop("temp_idx")

    print(f"Step 2: {len(env_list_passed)} / {len(env_list)} envs passed compilation test. Finished.")
    print(f"Step 2: {len(passed_code_id_map)} code_id passed compilation test. Finished.")

    # save into parquet or jsonl
    if target_file.endswith(".parquet"):
        list_of_dicts_to_parquet(dedup_env_list_passed, target_file)
    elif target_file.endswith(".jsonl"):
        with open(target_file, "w") as f:
            for item in dedup_env_list_passed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/codegym_generation_prompt_10_output.jsonl", help="the source file to read from")
    parser.add_argument("--target_file", type=str, default="example/codegym_envs.jsonl", help="the target file to save the envs")
    parser.add_argument("--force_mode", action="store_true", default=False, help="whether to force overwrite the target folder")
    parser.add_argument("--cutoff", type=int, default=10, help="the cutoff of the number of envs to save")
    parser.add_argument("--num_processes", type=int, default=32, help="the number of processes to use")
    args = parser.parse_args()
    
    # only merge into parquet
    # utils.merge_into_parquet(env_list_passed="envs/__env_list_passed.txt", target_folder="envs")

    main(input_file=args.input_file, target_file=args.target_file, force_mode=args.force_mode, cutoff=args.cutoff, num_processes=args.num_processes)