import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gym_to_docstring import gym_docstring_check

def generate_docstring_prompt(gym_env: str):
# generate docstring for SOLVE_FC
    flag, results = gym_docstring_check(gym_env)
    assert flag, f"{gym_docstring_check(gym_env, verbose=True)}"

    docstring_prompt = ""
    for label, doc in results.items():
        first_line = "def " + label + gym_env.split(f"def {label}")[1].split("\n")[0].strip()
        doc_lines = doc.split("\n")
        for i in range(len(doc_lines)):
            if doc_lines[i].startswith(" " * 8):
                doc_lines[i] = doc_lines[i][4:]
        docstring_prompt += "Function:\n\n" + first_line + "\n    r\"\"\"\n" + "\n".join(doc_lines) + "\n    \"\"\"\n\n"

    return docstring_prompt

with open("prompt/example_gym_env.py", "r") as f:
    EXAMPLE_GYM_ENV = f.read().strip()
with open("prompt/example_gym_task.txt", "r") as f:
    EXAMPLE_GYM_TASK = f.read().strip()
with open("prompt/solve_fc_task_description.txt", "r") as f:
    SOLVE_FC_TASK_DESCRIPTION = f.read().strip()

EXAMPLE_GYM_ENV_WO_SOLVE = EXAMPLE_GYM_ENV.split("def solve(")[0].strip()
EXAMPLE_SOLVE_FC = ("    def solve(" + EXAMPLE_GYM_ENV.split("def solve(")[1].split("if __name__ == \"__main__\"")[0])

while EXAMPLE_SOLVE_FC.split("\n")[-1].strip().startswith("#") or len(EXAMPLE_SOLVE_FC.split("\n")[-1].strip()) == 0:
    EXAMPLE_SOLVE_FC = "\n".join(EXAMPLE_SOLVE_FC.split("\n")[:-1])

# remove first 4 spaces in each line of SOLVE_FC
EXAMPLE_SOLVE_FC = "\n".join([line[4:] for line in EXAMPLE_SOLVE_FC.split("\n")])

while EXAMPLE_GYM_ENV_WO_SOLVE.split("\n")[-1].strip().startswith("#") or len(EXAMPLE_GYM_ENV_WO_SOLVE.split("\n")[-1].strip()) == 0:
    EXAMPLE_GYM_ENV_WO_SOLVE = "\n".join(EXAMPLE_GYM_ENV_WO_SOLVE.split("\n")[:-1])

EXAMPLE_DOCSTRING_PROMPT = generate_docstring_prompt(EXAMPLE_GYM_ENV_WO_SOLVE)

SOLVE_FC_PROMPT = """# 任务描述

{task_description}

## 示例问题及答案

### 输入

#### 问题场景
{example_gym_task}

#### 环境
{example_docstring_prompt}

### 输出

<answer>
{example_answer}
</answer>

## 问题：

### 输入

#### 问题场景
{gym_task}

#### 环境
{docstring_prompt}

### 输出


"""
# .format(task_description=SOLVE_FC_TASK_DESCRIPTION, example_gym_task=EXAMPLE_GYM_TASK, example_docstring_prompt=EXAMPLE_DOCSTRING_PROMPT, example_answer=EXAMPLE_SOLVE_FC)
