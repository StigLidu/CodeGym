# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

from collections import defaultdict

with open("prompt_en/example_code_problem_description.txt", "r") as f:
    EXAMPLE_CODE_PROBLEM_DESCRIPTION = f.read().strip()
with open("prompt_en/example_code_solution.txt", "r") as f:
    EXAMPLE_CODE_SOLUTION = f.read().strip()
with open("prompt_en/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()
with open("prompt_en/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt_en/task_description.txt", "r") as f:
    TASK_DESCRIPTION = f.read().strip()

class SafeDict(defaultdict):
    def __missing__(self, key):
        return '{' + key + '}'

GYM_GEN_PROMPT = """You are an expert skilled at transforming code problems into interactive environments.

# Task Description

{task_description}

## Example

### Input

<problem>
{example_problem_description}
</problem>

<code>
{example_solution_code}
</code>

### Output

<task>
{example_gym_task}
</task>

<env>
{example_gym_env}
</env>

# Requirements Restatement

{task_description}

Transform the following problem and code:

### Input

<problem>
{problem_description}
</problem>

<code>
{solution_code}
</code>

### Your Output
"""
