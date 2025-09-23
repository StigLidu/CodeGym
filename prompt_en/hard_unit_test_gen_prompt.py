import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("prompt_en/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt_en/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()

UNIT_TEST_GENERATION_PROMPT = """
## Task Description

You are an intelligent assistant responsible for generating unit test cases for Python functions based on problem descriptions and gym environments. You will be given a problem description and a gym environment, and your task is to generate 10 relatively difficult test cases for this environment.

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

* **Definition of Difficulty**

  * Parameter values in test cases can be relatively large, such as large array lengths, large numerical values, etc.
  * Solving this requires relatively complex gym environment logic, such as using multiple functions or relatively complex calling logic, etc.

### Problem Description

<problem_description>
{problem_description}
</problem_description>

### Gym Environment

<gym_env>
{gym_env}
</gym_env>

Some simple test cases are as follows:

<simple_test_case>
{simple_test_case}
</simple_test_case>

Please refer to but do not copy these simple test cases. Please output one test case per line, reiterating that the test format is `a@b`, where `a` is the environment class name and `b` is the parameter dictionary passed in, written in valid JSON format (e.g., `{{"arg1": [...]}}`).

"""
