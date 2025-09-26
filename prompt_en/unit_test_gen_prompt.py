# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("prompt_en/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt_en/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()

UNIT_TEST_GENERATION_PROMPT = """
## Task Description

You are an intelligent assistant responsible for generating unit test cases for Python functions based on problem descriptions and gym environments. You will be given a problem description and a gym environment, and your task is to generate 15 test cases for this environment.

Please ensure all test cases follow these requirements:

* **Input must be valid JSON strings**:

  * No Python expressions allowed (such as `[1]*5` or `[i%11 for i in range(100)]`)
  * Must not contain comments, computational expressions, or Python syntax sugar

* **Each test case must use the `a@b` format, where:**

  * `a` is the environment class name
  * `b` is the parameter dictionary passed in, written in valid JSON format (e.g., `{{"arg1": [...]}}`)
  * Example: `ModeFindingEnv@{{"scores": [1, 2, 9, 6, 10, 4, 1, 5, 8, 8, 2, 10, 1, 3, 8, 0, 0, 5, 3, 5]}}`

* **Test cases need to cover various situations**, including typical and boundary cases:

  * Different sizes, diverse structures, different value distributions, etc.

* **Arranged by increasing difficulty**:

  * First 5 are easy difficulty
  * Middle 5 are medium difficulty
  * Last 5 are hard difficulty (need to cover extreme or boundary cases)

### Problem Description

<problem_description>
{problem_description}
</problem_description>

### Gym Environment

<gym_env>
{gym_env}
</gym_env>

There may be some unit tests in the main function of the environment that do not follow the format I want for generating unit tests. You can refer to these unit tests, but please do not copy them completely.

Please output one unit test per line, reiterating that the test format is `a@b`, where `a` is the environment class name and `b` is the parameter dictionary passed in, written in valid JSON format (e.g., `{{"arg1": [...]}}`).

"""


test_example = """Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_.

You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice.

You can return the answer in any order.

**Example 1:**

**Input:** nums = \[2,7,11,15\], target = 9
**Output:** \[0,1\]
**Explanation:** Because nums\[0\] + nums\[1\] == 9, we return \[0, 1\].

**Example 2:**

**Input:** nums = \[3,2,4\], target = 6
**Output:** \[1,2\]

**Example 3:**

**Input:** nums = \[3,3\], target = 6
**Output:** \[0,1\]

**Constraints:**

* `2 <= nums.length <= 104`
* `-109 <= nums[i] <= 109`
* `-109 <= target <= 109`
* **Only one valid answer exists.**

**Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?"""
