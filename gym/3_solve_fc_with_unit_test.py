import sys
import os
import shutil
import types
import numpy as np
import queue
from tqdm import tqdm
import argparse
import time
import tracemalloc
import multiprocessing as mp
import traceback
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_jsonl_or_parquet

def do_unit_test_solve_fc(env_code, env_name, unit_test, max_mem, debug_mode=False):
    # WARNING: timeout assertation should be in the main function with multiprocessing
    try:
        tracemalloc.start()
        start_time = time.time()

        module_name = f"temp_env_{env_name}"
        temp_module = types.ModuleType(module_name)
        exec(env_code, temp_module.__dict__)
        env_class = getattr(temp_module, env_name)

        env = env_class.from_env_str(unit_test)
        output = env.solve()
        step_count = env.step_count

        elapsed_time = time.time() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if peak_mem > max_mem:
            return None

        if debug_mode:
            print("[INPUT]", unit_test)
            print("[OUTPUT]", output)
            print("[TIME]", elapsed_time)
            print("[MEM]", peak_mem)
            print("-" * 60)

        return {
            "output": output,
            "solve_fc_round": step_count
        }

    except Exception as e:
        if debug_mode:
            traceback.print_exc()
            print(f"[ERROR] Failed on unit_test: {unit_test} - {e}")
        return None

def _mp_target(conn, func, args, kwargs):
    try:
        out = func(*args, **(kwargs or {}))
        conn.send(("ok", out))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()

def run_task_with_timeout(func, args=(), kwargs=None, timeout=5):
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_mp_target, args=(child_conn, func, args, kwargs))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None

    if parent_conn.poll():
        status, payload = parent_conn.recv()
        return payload if status == "ok" else None
    return None

# ----------------- worker -----------------
def worker(task_q, result_q, timeout_per_task, max_mem, debug_mode):
    while True:
        tc = task_q.get()
        if tc is None:
            time.sleep(0.5)
            continue
        idx1, idx2 = tc["solve_fc_idx"], tc["ut_idx"]
        res = run_task_with_timeout(
            do_unit_test_solve_fc,
            args=(tc["gym_env"], tc["env_name"], tc["unit_test"], max_mem, debug_mode),
            timeout=timeout_per_task,
        )
        result_q.put((idx1, idx2, res))

# ----------------- main process -----------------
def run_all_tests_parallel(
    test_cases,
    ans_collection,
    task_q,
    result_q,
    global_timeout=None,
    timeout=2
):
    start_time = time.time()

    for tc in test_cases:
        task_q.put(tc)

    remaining = len(test_cases)

    while remaining > 0:
        if global_timeout is not None and (time.time() - start_time) > global_timeout:
            print("[Global Timeout] Stop waiting.")
            break
        try:
            idx1, idx2, res = result_q.get(timeout=0.5)
        except Exception as e:
            continue

        if (res is None) or (res.get("output") is None) or ("reward=1" not in res["output"]):
            ans_collection[idx1, idx2] = -1
        else:
            ans_collection[idx1, idx2] = res["solve_fc_round"]

        remaining -= 1

    # clean up task_q and result_q
    if remaining > 0:
        try:
            while True:
                task_q.get_nowait()
        except queue.Empty:
            pass
        time.sleep(timeout)
        try:
            while True:
                result_q.get_nowait()
        except queue.Empty:
            pass

def main(unit_test_file, solve_fc_file, target_folder, force_mode=False, debug_mode=False, timeout=2, max_mem=104857600, num_processes=128, resume_mode=False):
    # load unit test file
    df_unit_test = load_jsonl_or_parquet(unit_test_file)
    df_solve_fc = load_jsonl_or_parquet(solve_fc_file)

    if os.path.exists(target_folder) and force_mode:
        shutil.rmtree(target_folder)

    passed_code_id_list = []
    if not resume_mode:
        os.makedirs(target_folder, exist_ok=False)
    else:
        # list all files in the target folder
        file_list = os.listdir(target_folder)
        passed_code_id_list = sorted([f.split("__")[0] for f in file_list if f.endswith(".json")])
        print(f"Resume from {passed_code_id_list[-1]}")

    # merge the two dataframes
    df_unit_test = sorted(df_unit_test, key=lambda x: (x["code_id"], x["idx"]))
    df_solve_fc = sorted(df_solve_fc, key=lambda x: (x["code_id"], x["idx"]))

    # start all workers
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    manager = mp.Manager()
    task_q = manager.Queue()
    result_q = manager.Queue()
    workers = [
        mp.Process(target=worker, args=(task_q, result_q, timeout, max_mem, debug_mode))
        for _ in range(num_processes)
    ]
    for p in workers:
        p.start()

    j = 0
    with tqdm(total=len(df_unit_test), desc="Unit testing") as pbar:
        for i in range(len(df_unit_test)):
            pbar.update(1)
            if len(passed_code_id_list) > 0 and (df_unit_test[i]["code_id"] in passed_code_id_list or df_unit_test[i]["code_id"] < passed_code_id_list[-1]):
                continue
            pbar.set_postfix(passed=len(passed_code_id_list), failed = i - len(passed_code_id_list))
            # find the corresponding solve_fc
            while j < len(df_solve_fc) and (df_solve_fc[j]["code_id"] < df_unit_test[i]["code_id"] or (df_solve_fc[j]["code_id"] == df_unit_test[i]["code_id"] and df_solve_fc[j]["idx"] < df_unit_test[i]["idx"])):
                j += 1
            if j >= len(df_solve_fc) or df_solve_fc[j]["code_id"] != df_unit_test[i]["code_id"] or df_solve_fc[j]["idx"] != df_unit_test[i]["idx"]:
                if debug_mode:
                    print(f"The unit test {i} cannot pass the corresponding solve_fc because the solve_fc is not found")
                continue
            k = j
            while k < len(df_solve_fc) and (df_solve_fc[k]["code_id"] == df_unit_test[i]["code_id"] and df_solve_fc[k]["idx"] == df_unit_test[i]["idx"]):
                k += 1

            # left is j, right is k - 1
            # we have k - j solve_fc to check
            # multiprocessing k - j * all unit_test_list

            unit_test_list = df_unit_test[i]["unit_test_list"]
            test_cases = []
            for solve_fc_idx, solve_fc in enumerate(df_solve_fc[j:k]):
                for ut_idx, ut in enumerate(unit_test_list):
                    test_cases.append({
                        "gym_env": solve_fc["gym_env"],
                        "env_name": solve_fc["env_name"][:-3] if solve_fc["env_name"].endswith(".py") else solve_fc["env_name"],
                        "unit_test": ut["unit_test"],
                        "solve_fc_idx": solve_fc_idx,
                        "ut_idx": ut_idx
                    })

            ans_collection = np.zeros((k - j, len(unit_test_list))).astype(int)
            ans_collection.fill(0)

            try:
                run_all_tests_parallel(
                    test_cases,
                    ans_collection,
                    task_q,
                    result_q,
                    global_timeout=15,
                    timeout=timeout
                )
            except Exception as e:
                if debug_mode:
                    print(f"[ERROR] {e}")
                continue

            for solve_fc_idx in range(k - j):
                if np.all(~np.isin(ans_collection[solve_fc_idx], (-1, 0))):
                    passed_code_id_list.append(df_solve_fc[j + solve_fc_idx]["code_id"])
                    print(f"The solve_fc {j + solve_fc_idx} can pass the corresponding unit test, solved env count {len(passed_code_id_list)}")
                    print("-" * 100)
                    for ut_idx in range(len(unit_test_list)):
                        unit_test_list[ut_idx]["solve_fc_round"] = int(ans_collection[solve_fc_idx, ut_idx])
                    df_unit_test[i]["gym_env"] = df_solve_fc[j + solve_fc_idx]["gym_env"]
                    df_unit_test[i]["docstring"] = df_solve_fc[j + solve_fc_idx]["docstring"]
                    df_unit_test[i]["unit_test_list"] = unit_test_list
                    with open(os.path.join(target_folder, f"{df_solve_fc[j + solve_fc_idx]['code_id']}__{df_solve_fc[j + solve_fc_idx]['env_name']}.json"), "w") as f:
                        json.dump(df_unit_test[i], f, ensure_ascii=False)
                    break

    print(f"Total solved env count: {len(passed_code_id_list)}")

    # graceful worker and manager cleanup, may take a while
    try:
        # signal workers to stop (workers currently ignore None, but keep for future behavior)
        for _ in workers:
            try:
                task_q.put(None)
            except Exception:
                break

        # attempt graceful join
        for p in workers:
            try:
                p.join(timeout=0.05)
            except Exception:
                pass

        # force terminate any remaining workers
        for p in workers:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join()
                except Exception:
                    pass

        # close queues
        try:
            task_q.close(); task_q.join_thread()
        except Exception:
            pass

        try:
            result_q.close(); result_q.join_thread()
        except Exception:
            pass

        # shutdown manager
        try:
            manager.shutdown()
        except Exception:
            pass
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test_file", type=str, default="example/codegym_unit_test.jsonl", help="the input file to read from")
    parser.add_argument("--solve_fc_file", type=str, default="example/codegym_solve_fc.jsonl", help="the input file to read from")
    parser.add_argument("--target_folder", type=str, default="example/codegym_env_verified", help="the target folder to save the envs")
    parser.add_argument("--force_mode", action="store_true", default=False, help="whether to force overwrite the target folder")
    parser.add_argument("--resume_mode", action="store_true", default=False, help="whether to resume from the target folder")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="whether to print debug information")
    parser.add_argument("--timeout", type=int, default=3, help="the timeout for the unit test")
    parser.add_argument("--max_mem", type=int, default=104857600, help="the max memory for the unit test")
    parser.add_argument("--num_processes", type=int, default=128, help="the number of processes to use")
    args = parser.parse_args()
    assert not (args.force_mode and args.resume_mode), "force_mode and resume_mode cannot be set at the same time"
    main(unit_test_file=args.unit_test_file, \
        solve_fc_file=args.solve_fc_file, \
        target_folder=args.target_folder, \
        force_mode=args.force_mode, \
        debug_mode=args.debug_mode, \
        timeout=args.timeout, \
        max_mem=args.max_mem, \
        num_processes=args.num_processes, \
        resume_mode=args.resume_mode)