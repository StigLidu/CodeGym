# remove `self` in each action
# two actions: `Observe` and `Done`, make them more diverse, i.e., paraphrase

import json
import sys
import random

def remove_self_and_example_for_prompt(prompt: str):
    prompt_lines = prompt.split("\n")
    new_prompt_lines = []
    example_flag = False
    for line in prompt_lines:
        if line.startswith("def") and "(self, " in line:
            line = line.replace("self, ", "")
        if line.startswith("def") and "(self)" in line:
            line = line.replace("self", "")
        assert "self" not in line, line
        if "Example Output" in line:
            example_flag = True
        if "\"\"\"" in line:
            example_flag = False
        if not example_flag:
            new_prompt_lines.append(line)
    prompt = "\n".join(new_prompt_lines)
    return prompt

def remove_self(jsonl_file_path):
    all_data = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            docstring = data["prompt"][0]["content"]
            docstring_lines = docstring.split("\n")
            new_docstring_lines = []
            for line in docstring_lines:
                if line.startswith("def") and "(self, " in line:
                    line = line.replace("self, ", "")
                if line.startswith("def") and "(self)" in line:
                    line = line.replace("self", "")
                assert "self" not in line, line
                new_docstring_lines.append(line)
            docstring = "\n".join(new_docstring_lines)
            data["prompt"][0]["content"] = docstring
            all_data.append(data)

    with open(jsonl_file_path.replace(".jsonl", "_rm_self.jsonl"), "w") as f:
        for line in all_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

OBSERVE_ACTION_ALTERNATIVES = [
    "Observe",
    "GetObs",
    "GetObservation",
    "FetchObs",
    "FetchObservation",
    "AcquireObs",
    "AcquireObservation",
    "ReadObs",
    "ReadObservation",
    "CaptureObs",
    "CaptureObservation",
    "CollectObs",
    "CollectObservation",
    "GrabObs",
    "GrabObservation",
    "PullObs",
    "PullObservation",
    "Sense",
    "SenseEnv",
    "Perceive",
    "ScanEnv"
]

DONE_FUNC_ALTERNATIVES = [
    "Done",
    "Finish",
    "Complete",
    "End",
    "Stop",
    "Submit",
    "SubmitAns",
    "SubmitAnswer",
    "Finalize",
    "Final",
    "Terminate",
    "Conclude",
    "Close",
    "WrapUp",
    "Result",
    "Answer",
    "Output",
    "Deliver",
    "TurnIn",
    "Accomplish"
]

# TODO: need to make sure the action is not in ALL the docstring

def random_action(jsonl_file_path):
    all_data = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            used_names = set()
            data = json.loads(line)
            docstring = data["prompt"][0]["content"]
            docstring_lines = docstring.split("\n")
            for line in docstring_lines:
                if line.startswith("def"):
                    used_names.add(line.split("(")[0].split(" ")[1])
            
            used_names.discard("Done")
            used_names.discard("Observe")

            alternative_obs = random.choice(OBSERVE_ACTION_ALTERNATIVES)
            alternative_done = random.choice(DONE_FUNC_ALTERNATIVES)

            while alternative_obs in used_names:
                alternative_obs = random.choice(OBSERVE_ACTION_ALTERNATIVES)
            while alternative_done in used_names:
                alternative_done = random.choice(DONE_FUNC_ALTERNATIVES)

            docstring = docstring.replace("def Observe(", f"def {alternative_obs}(")
            docstring = docstring.replace("def Done(", f"def {alternative_done}(")
            data["prompt"][0]["content"] = docstring
            all_data.append(data)

    with open(jsonl_file_path.replace(".jsonl", "_rd_action.jsonl"), "w") as f:
        for line in all_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
