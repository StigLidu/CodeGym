# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
import json
import sys
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.docstrong_format import remove_self_and_example_for_prompt

USER_PROMPT_CN = """请按下列要求一步步回答下面的问题！
1. 禁止编写代码来回答用户的问题，只能调用提供的函数, 每一步最多调用一个函数
2. 当你调用函数后，请等待工具返回结果，不要自己假设返回结果。
3. 如果工具描述不够清楚，你可以尝试用一下，通过拿到的结果来修正之前的工具调用。
4. 函数调用请用<|FunctionCallBegin|>....<|FunctionCallEnd|>包裹住一个json格式的list，这个list里面包含一个dict，其中每个dict包含两个参数一个是name表示函数名，一个是parameters表示参数。这里是一个函数调用的例子：<|FunctionCallBegin|>[{{\"name\":\"function_name\", \"parameters\":{{\"key1\":\"value1\", \"key2\":\"value2\"}}}}]<|FunctionCallEnd|>
额外的要求: 请不要过度思考，而是稍加思考后你就可以决定该怎么调用函数！并且由于你有很多轮调用函数的机会，没有必要从一开始就规划好每一步该调用什么函数。并且不要思考不用工具就解决问题！

问题：

"""

USER_PROMPT_EN = """Please answer the following question step by step according to the requirements below!

1. **Do not** write code to answer the user's question — you may only call the provided functions, and you may call at most **one function per step**.
2. After you call a function, wait for the tool to return the result — do not assume what the result will be.
3. If the tool’s description is unclear, you can try using it first, and then adjust your function call based on the returned result.
4. Function calls should be wrapped with `<|FunctionCallBegin|>....<|FunctionCallEnd|>` and contain a JSON-formatted list. The list should include **one dictionary**, where each dictionary contains two parameters:

   * `name`: the function name
   * `parameters`: a dictionary of key-value pairs for the arguments

Here’s an example of a function call:
<|FunctionCallBegin|>[{"name":"function_name", "parameters":{"key1":"value1", "key2":"value2"}}]<|FunctionCallEnd|>

**Extra requirements:**

* Do not overthink; think briefly, then decide how to call the function.
* Since you have many chances to call functions, you do not need to plan all steps in advance.
* Do not try to solve the problem without using the tools.

Question:

"""

def main(env_source_list, env_target_dir, training_data_file, min_solve_fc_round=10, max_solve_fc_round=256, force_mode=False, lang="EN"):
    os.makedirs(env_target_dir, exist_ok=True if force_mode else False)
    if os.path.exists(training_data_file):
        if force_mode:
            os.remove(training_data_file)
        else:
            assert False, f"Training data file {training_data_file} already exists, use --force_mode to overwrite it"
    training_data_list = []
    total_envs = 0
    for env_source_file in tqdm(os.listdir(env_source_list)):
        if not env_source_file.endswith(".json"):
            continue
        with open(os.path.join(env_source_list, env_source_file), "r") as f:
            env_data = json.load(f)
            code_id = env_data["code_id"]
            env_name = env_data["env_name"]

            #step 1: copy unit test list into training_data
            unit_test_list = env_data["unit_test_list"]
            gym_docstring = env_data["docstring"]
            gym_task = env_data["gym_task"]
            data_flag = False
            for unit_test in unit_test_list:
                ability_str = f"codegym_v1@{code_id}__{unit_test['unit_test']}"
                if unit_test["solve_fc_round"] < min_solve_fc_round or unit_test["solve_fc_round"] > max_solve_fc_round:
                    continue
                data_flag = True
                prompt = [{
                    'role': 'system',
                    'content': remove_self_and_example_for_prompt(gym_docstring),
                    'name': 'functions',
                }, {
                    'role': 'user',
                    'content': USER_PROMPT_CN if lang == "CN" else USER_PROMPT_EN + gym_task,
                    'name': '',
                }]
                training_data = {
                    'ability': ability_str,
                    'code_id': code_id,
                    'data_source': 'codegym_v1',
                    'prompt': prompt,
                    'reward_model': {
                        'ground_truth': None,
                        'style': 'agent_env',
                    },
                    "answer": "There is no answer", # TODO: actually we have the ref_answer in unit_test
                    "plugin_call_max_round": max_solve_fc_round,
                    "extra_info": {
                        "index": str(len(training_data_list)),
                        "gym_env": env_data["gym_env"],
                        "solve_fc_round": unit_test["solve_fc_round"],
                    }
                }
                training_data_list.append(training_data)
                with open(training_data_file, "a") as f:
                    f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
            
            if data_flag:
                total_envs += 1
            
                #step 2: copy "gym_env" (a code string) to env_target_dir
                gym_env_code = env_data["gym_env"]
                gym_env_path = os.path.join(env_target_dir, f"{code_id}__{env_name}.py")
                with open(gym_env_path, "w") as f:
                    f.write(gym_env_code)

    print(f"Total envs: {total_envs}")
    print(f"Total training data count: {len(training_data_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_source_list", type=str, default="example/codegym_env_verified", help="the source folder to read from")
    parser.add_argument("--env_target_dir", type=str, default="example/codegym_synth_env", help="the target folder to save the envs")
    parser.add_argument("--training_data_file", type=str, default="example/codegym_training_data.jsonl", help="the file to save the training data")
    parser.add_argument("--min_solve_fc_round", type=int, default=10, help="the minimum solve_fc_round")
    parser.add_argument("--max_solve_fc_round", type=int, default=256, help="the maximum solve_fc_round")
    parser.add_argument("--force_mode", action="store_true", default=False, help="whether to force overwrite the training data file")
    parser.add_argument("--lang", type=str, default="EN", help="the language of the training data")
    args = parser.parse_args()
    main(env_source_list=args.env_source_list, \
         env_target_dir=args.env_target_dir, \
         training_data_file=args.training_data_file, \
         min_solve_fc_round=args.min_solve_fc_round, \
         max_solve_fc_round=args.max_solve_fc_round, \
         force_mode=args.force_mode, \
         lang=args.lang)