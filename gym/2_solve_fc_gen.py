import os
import json
import sys
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_en.solve_fc_gen_prompt import *
from utils.online_inference import online_inference
from utils.utils import load_jsonl_or_parquet
from utils.parquet_utils import list_of_dicts_to_parquet

def main(data_path, save_path):
    data_list = load_jsonl_or_parquet(data_path)

    solve_fc_prompt_list = []
    for data in tqdm(data_list):
        gym_task = data["gym_task"]
        gym_env = data["gym_env"]
        docstring_prompt = generate_docstring_prompt(gym_env)
        solve_fc = SOLVE_FC_PROMPT.format(
            task_description=SOLVE_FC_TASK_DESCRIPTION,
            example_gym_task=EXAMPLE_GYM_TASK,
            example_docstring_prompt=EXAMPLE_DOCSTRING_PROMPT,
            example_answer=EXAMPLE_SOLVE_FC,
            gym_task=gym_task,
            docstring_prompt=docstring_prompt
        )
        solve_fc_prompt_list.append({
            "code_id": data["code_id"],
            "idx": data["idx"],
            "env_name": data["env_name"],
            "gym_task": gym_task,
            "gym_env": gym_env,
            "docstring_prompt": docstring_prompt,
            "solve_fc_prompt": solve_fc
        })

        # print the first solve_fc_prompt as an example
        if len(solve_fc_prompt_list) == 1:
            print(solve_fc_prompt_list[0]["solve_fc_prompt"])

    # save into parquet or jsonl
    if save_path.endswith(".parquet"):
        list_of_dicts_to_parquet(solve_fc_prompt_list, save_path)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for item in solve_fc_prompt_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported data format: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/codegym_envs.jsonl", help="the input file to read from")
    parser.add_argument("--output_file", type=str, default="example/codegym_solve_fc_prompt.jsonl", help="the output file to save to")
    parser.add_argument("--online-inference", action="store_true", default=False, help="whether to use online inference")
    args = parser.parse_args()
    main(data_path=args.input_file, save_path=args.output_file)
    if args.online_inference:
        online_inference(args.output_file, input_key="solve_fc_prompt", output_key="output", num = 10, repeat = 1)