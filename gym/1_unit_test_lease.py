import json
import os
import sys
import shutil
from tqdm import tqdm
import time
import tracemalloc
import types
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_jsonl_or_parquet

def do_unit_test_ref_answer(env_code, env_name, unit_test, max_mem, max_ans_len, debug_mode=False):
    # WARNING: timeout assertation should be in the main function with multiprocessing
    try:
        tracemalloc.start()
        start_time = time.time()

        module_name = f"temp_env_{env_name}"
        temp_module = types.ModuleType(module_name)
        exec(env_code, temp_module.__dict__)
        env_class = getattr(temp_module, env_name)

        env = env_class.from_env_str(unit_test)
        answer = env.get_ref_answer()
        elapsed_time = time.time() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if len(str(answer)) > max_ans_len or peak_mem > max_mem:
            return None

        return {
            "unit_test": unit_test,
            "ref_answer": answer,
            "time": elapsed_time,
            "mem": peak_mem
        }

    except Exception as e:
        if debug_mode:
            traceback.print_exc()
            print(f"[ERROR] Failed on unit_test: {unit_test} - {e}")
        return None

def main(source_file, target_file, force_mode=False, debug_mode=False, timeout=2, max_mem=104857600, max_ans_len=100, num_processes=32):
    if os.path.exists(target_file) and force_mode:
        os.remove(target_file)

    if os.path.exists(target_file):
        raise FileExistsError(f"Target file {target_file} already exists")

    total_env_count = 0
    saved_code_id_set = set()

    data_list = load_jsonl_or_parquet(source_file)

    # get total lines first
    total_lines = sum(1 for _ in data_list)

    for data in tqdm(data_list, total=total_lines, desc="Processing"):
        env_name = data["env_name"]
        code_id = data["code_id"]

        # step 0: check if the env has been saved
        if code_id in saved_code_id_set:
            continue

        if env_name.endswith(".py"):
            env_name = env_name[:-3]

        """
        # Step 1: dynamically load env class from data["gym_env"]
        env_code = data["gym_env"]
        module_name = f"temp_env_{code_id}_{data['idx']}"
        temp_module = types.ModuleType(module_name)
        exec(env_code, temp_module.__dict__)
        env_class = getattr(temp_module, env_name)
        """

        all_passed_unit_test_list = []
        unit_test_list = []

        for i, output in enumerate(data["output"]):
            if "</think>" in output:
                output = output.split("</think>")[1]

            for line in output.split("\n"):
                # extract the unit test from the output, the format is: env_name@unit_test
                if f"{env_name}@" not in line:
                    continue
                line = env_name + "@" + line.split(f"{env_name}@")[1]
                if not line.startswith(f"{env_name}@"):
                    continue
                params = line.split("@")[1].strip()
                try:
                    params = json.loads(params)
                except Exception as e:
                    if debug_mode:
                        print(f"[ERROR] {line} -> {e}")
                    continue
                unit_test_list.append(line.strip())

        with Pool(processes=num_processes) as pool:
            tasks = [pool.apply_async(do_unit_test_ref_answer, (data["gym_env"], env_name, ut, max_mem, max_ans_len, debug_mode)) for ut in unit_test_list]
            for task in tqdm(tasks, total=len(unit_test_list), desc="Unit testing"):
                try:
                    result = task.get(timeout=timeout)
                    if result is not None:
                        all_passed_unit_test_list.append(result)
                except Exception as e:
                    if debug_mode:
                        print(f"[ERROR] {task.args} -> {e}")
                    continue

        if debug_mode:
            print(f"[TOTAL_UNIT_TEST_COUNT] {env_name}: {len(unit_test_list)}")
            print(f"[ALL_PASSED_UNIT_TEST_LIST] {env_name}: {len(all_passed_unit_test_list)}")
            print("-" * 100)

        if len(all_passed_unit_test_list) < len(unit_test_list) * 0.8 or len(all_passed_unit_test_list) < 10:
            # The env may be buggy, skip it
            # The total unit test count should be 2 x 15 = 30, but some outputs from LLM are not complete
            # TODO: Make it configurable
            continue

        # Save the env, should be: [code_id], [idx], [env_name], [gym_env], [gym_task], [unit_test_list]
        env_with_unit_test_list = {
            "code_id": data["code_id"],
            "idx": data["idx"],
            "env_name": env_name,
            "gym_env": data["gym_env"],
            "gym_task": data["gym_task"],
            "unit_test_list": all_passed_unit_test_list
        }

        # save the env with valid unit tests
        try:
            with open(target_file, "a") as f:
                f.write(json.dumps(env_with_unit_test_list, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            print(f"[ERROR] {env_name} cannot be saved, error: {e}")
            continue

        total_env_count += 1
        saved_code_id_set.add(code_id)

        if debug_mode:
            print(f"[SAVED] {env_name}, unit_test_count: {len(env_with_unit_test_list['unit_test_list'])}; total_env_count: {total_env_count}")
            print("-" * 100)

    print(f"[TOTAL_SAVED_ENV_COUNT] {total_env_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="example/codegym_unit_test_prompt_output.jsonl", help="the input file to read from")
    parser.add_argument("--target_file", type=str, default="example/codegym_unit_test.jsonl", help="the target file to save the envs")
    parser.add_argument("--force_mode", action="store_true", default=False, help="whether to force overwrite the target folder")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="whether to print debug information")
    parser.add_argument("--timeout", type=int, default=3, help="the timeout for the unit test")
    parser.add_argument("--max_mem", type=int, default=104857600, help="the max memory for the unit test")
    parser.add_argument("--max_ans_len", type=int, default=100, help="the max length of the ref answer")
    parser.add_argument("--num_processes", type=int, default=32, help="the number of processes to use")
    args = parser.parse_args()
    main(
        source_file=args.input_file, \
        target_file=args.target_file, \
        force_mode=args.force_mode, \
        debug_mode=args.debug_mode, \
        timeout=args.timeout, \
        max_mem=args.max_mem, \
        max_ans_len = args.max_ans_len, \
        num_processes=args.num_processes \
    )