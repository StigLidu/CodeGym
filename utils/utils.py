import os
import subprocess
import tempfile
import json
import pandas as pd
import io
import contextlib
from utils.gym_to_docstring import gym_docstring_check
import io
import contextlib
import traceback
import multiprocessing
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import re
import ast

def pass_fc_from_output(output: str, fc_name: str = None, indent_level: int = 0):
    if fc_name is None:
        prefix = "def "
    else:
        prefix = f"def {fc_name}("

    # find the first line that starts with "def "
    start_line = None
    end_line = None
    for idx, line in enumerate(output.split("\n")):
        if start_line is not None and not line.startswith(" " * (indent_level + 1)) and not line.strip() == "":
            end_line = idx
            break
        if line.startswith(" " * indent_level + prefix):
            start_line = idx

    if start_line is None or end_line is None:
        return None, None, None

    code_block = "\n".join(output.split("\n")[start_line:end_line])

    # special case for seed 1.6
    if "</think>" in code_block:
        code_block = code_block.split("</think>")[0]

    return code_block, start_line, end_line

def add_indent(code_block: str, indent_level: int):
    return "\n".join([" " * indent_level + line for line in code_block.split("\n")])

def check_env(env, check_correctness=False, check_docstring=True, verbose=False, timeout=5, workspace="../workspace/"):
    if not os.path.exists(env):
        if not os.path.exists(workspace):
            os.makedirs(workspace, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir=workspace)
        temp_file_path = temp_file.name
        with open(temp_file_path, "w") as f:
            f.write(env)
    else:
        temp_file_path = env

    try:
        result = subprocess.run(
            ["python3", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True  # Python 3.7+; for earlier, use encoding='utf-8'
        )
        exit_code = result.returncode
        stdout_str = result.stdout
        stderr_str = result.stderr

    except subprocess.TimeoutExpired:
        stdout_str = ''
        stderr_str = 'Timeout'
        exit_code = -1

    if not os.path.exists(env):
        os.remove(temp_file_path)

    # Compilation Check
    if exit_code != 0:
        if verbose:
            print(f"[ERROR] failed compilation test. {stderr_str}")
        return False

    # Correctness Check
    if check_correctness:
        if stdout_str.count("reward=1") < 2 or stdout_str.count("reward=0") > 0:
            if verbose:
                print(f"[ERROR] failed correctness test. {stdout_str}")
            return False

    # Docstring Check
    if check_docstring:
        flag, _ = gym_docstring_check(env, verbose=verbose)
        if not flag:
            if verbose:
                print(f"[ERROR] failed docstring check.")
            return False
    
    if verbose:
        print(f"[INFO] passed env checks.")
    return True

def merge_into_parquet(env_list_passed, target_folder, save_env_dup = 1):
    if type(env_list_passed) == str:
        with open(env_list_passed, "r") as f:
            env_list_passed = [line.strip() for line in f]
    else:
        env_list_passed = [line.strip() for line in env_list_passed]

    env_list_passed = sorted(env_list_passed)

    # if code_id is the same, save up to save_env_dup envs
    env_list_passed_dict = {}
    for env_path in env_list_passed:
        code_id = env_path.split("__")[0].split("/")[-1]
        if code_id not in env_list_passed_dict:
            env_list_passed_dict[code_id] = []
        env_list_passed_dict[code_id].append(env_path)
    
    total_env_count = 0
    unique_env_count = 0
    saved_env_list_passed = []
    for code_id, env_paths in env_list_passed_dict.items():
        if len(env_paths) > save_env_dup and save_env_dup != -1:
            env_paths = env_paths[:save_env_dup]
        for env_path in env_paths:
            env_name = env_path.split("__")[-1]
            task = open(env_path.replace(".py", "__task.txt"), "r").read()
            env = open(env_path, "r").read()
            if os.path.exists(env_path.replace(".py", "__docstring.txt")):
                docstring = open(env_path.replace(".py", "__docstring.txt"), "r").read()
            else:
                docstring = ""
            saved_env_list_passed.append({
                "code_id": code_id,
                "idx": env_path.split("__")[1],
                "env_name": env_name,
                "gym_task": task,
                "gym_env": env
            })
            if docstring != "":
                saved_env_list_passed[-1]["docstring"] = docstring
        total_env_count += len(env_paths)
        unique_env_count += (len(env_paths) > 0)
    
    # save into parquet
    df = pd.DataFrame(saved_env_list_passed, columns=list(saved_env_list_passed[0].keys()))
    df.to_parquet(os.path.join(target_folder, "__envs.parquet"))
    print(f"total_env_count: {total_env_count}, unique_env_count: {unique_env_count}, saved_env_count: {len(saved_env_list_passed)}")

def read_parquet_as_records(file_path, batch_size=1000):
    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for record in df.to_dict(orient="records"):
            yield record

def load_jsonl_or_parquet(file_path):
    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith(".parquet"):
        return read_parquet_as_records(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def has_chinese(text: str) -> bool:
    """
    If text contains at least one Chinese character, return True, otherwise return False.
    """
    # Common Chinese Unicode range: \u4e00-\u9fff
    pattern = re.compile(r'[\u4e00-\u9fff]')
    matches = pattern.findall(text)
    return bool(matches)

def only_ascii_code_chars(text: str) -> bool:
    ascii_flag = [ord(c) < 128 for c in text]
    return all(ascii_flag)

def load_anything(data):
    try:
        return json.loads(data)
    except:
        pass
    if type(data) == str:
        try:
            return ast.literal_eval(data)
        except:
            pass
    return data