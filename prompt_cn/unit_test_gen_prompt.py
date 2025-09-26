# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("prompt/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()

UNIT_TEST_GENERATION_PROMPT = """
## 任务描述

你是一名智能助手，负责根据问题描述和 gym 环境为 Python 函数生成单元测试用例。你将获得一个问题描述和一个 gym 环境，任务是为该环境生成 15 个测试用例。

请确保所有测试用例遵守以下要求：

* **输入必须是合法的 JSON 字符串**：

  * 不允许使用任何 Python 表达式（如 `[1]*5` 或 `[i%11 for i in range(100)]`）
  * 不得包含注释、计算表达式或 Python 语法糖

* **每个测试用例必须采用 `a@b` 的格式，其中：**

  * `a` 为环境类的名称
  * `b` 为传入的参数字典，写成合法的 JSON 格式（如 `{{"arg1": [...]}}`）
  * 例子：`ModeFindingEnv@{{"scores": [1, 2, 9, 6, 10, 4, 1, 5, 8, 8, 2, 10, 1, 3, 8, 0, 0, 5, 3, 5]}}`

* **测试用例需覆盖各种情况**，包括典型情况和边界情况：

  * 大小不同、结构多样、数值分布不同等

* **按难度递增排列**：

  * 前 5 个为简单难度
  * 中间 5 个为中等难度
  * 后 5 个为困难难度（需覆盖极端或边界情况）

### 问题描述

<problem_description>
{problem_description}
</problem_description>

### gym 环境

<gym_env>
{gym_env}
</gym_env>

在环境的main函数中可能存在一些单元测试，他们并不遵守我想要的生成的单元测试的格式，你可以参考这些单元测试，但请务必不要完全照搬。

请每行输出一个单元测试，重申测试的格式为`a@b`，其中`a`为环境类的名称，`b`为传入的参数字典，写成合法的 JSON 格式（如 `{{"arg1": [...]}}`）。

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
