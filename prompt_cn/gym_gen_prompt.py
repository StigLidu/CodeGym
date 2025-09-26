# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

from collections import defaultdict

with open("prompt/example_code_problem_description.txt", "r") as f:
    EXAMPLE_CODE_PROBLEM_DESCRIPTION = f.read().strip()
with open("prompt/example_code_solution.txt", "r") as f:
    EXAMPLE_CODE_SOLUTION = f.read().strip()
with open("prompt/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()
with open("prompt/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt/task_description.txt", "r") as f:
    TASK_DESCRIPTION = f.read().strip()

class SafeDict(defaultdict):
    def __missing__(self, key):
        return '{' + key + '}'

GYM_GEN_PROMPT = """你是一位擅长将代码类问题转化为交互式环境的专家。

# 任务描述

{task_description}

## 示例

### 输入

<problem>
{example_problem_description}
</problem>

<code>
{example_solution_code}
</code>

### 输出

<task>
{example_gym_task}
</task>

<env>
{example_gym_env}
</env>

# 要求重申

{task_description}

针对如下问题和代码进行改造：

### 输入

<problem>
{problem_description}
</problem>

<code>
{solution_code}
</code>

### 你的输出
"""