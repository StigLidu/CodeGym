import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("prompt/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()

UNIT_TEST_GENERATION_PROMPT = """
## 任务描述

你是一名智能助手，负责根据问题描述和 gym 环境为 Python 函数生成单元测试用例。你将获得一个问题描述和一个 gym 环境，任务是为该环境生成 10 个比较困难的测试用例。

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

* **难的定义**

  * 出现在测试用例中的参数的值可以比较大，比如数组长度可以比较大，数值可以比较大，等等
  * 解决这个要用到比较复杂的 gym 环境的逻辑，比如需要用到多个函数，或者需要用到比较复杂的调用逻辑，等等

### 问题描述

<problem_description>
{problem_description}
</problem_description>

### gym 环境

<gym_env>
{gym_env}
</gym_env>

一些简单的测试用例如下：

<simple_test_case>
{simple_test_case}
</simple_test_case>

请参考但不要照搬这些简单的测试用例。请每行输出一个测试用例，重申测试的格式为`a@b`，其中`a`为环境类的名称，`b`为传入的参数字典，写成合法的 JSON 格式（如 `{{"arg1": [...]}}`）。

"""
