# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MinimumDeletionsEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.COMPARE_ADJACENT = 0
        self.COUNT_DELETIONS = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CompareAdjacentCharacters": self.COMPARE_ADJACENT,
            "CountDeletions": self.COUNT_DELETIONS,
            "Observe": self.OBSERVE,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property and staticmethod of the environment
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "MinimumDeletionsEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinimumDeletionsEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.n = len(self.s)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        deletions = 0
        for i in range(len(self.s) - 1):
            if self.s[i] == self.s[i + 1]:
                deletions += 1
        return deletions

    # [Required] Define the step method of the environment
    def step(self, action: str):
        r"""
        Execute an action, and return the result information. 

        Args:
            action (str): The JSON string of the action name and parameters. 

        Returns:
            tuple[bool, str]: Whether the action is executed successfully, and the result information. 
        """
        self.step_count += 1
        try:
            call_dict = json.loads(action)
            assert "name" in call_dict, "function call doesn't have `name`"
            assert "parameters" in call_dict, "function call doesn't have `parameters`"
            action_name = call_dict["name"]
            params = call_dict["parameters"]

            if action_name not in self.func_mapping:
                raise ValueError(f"Invalid action: {action_name}")
            
            action_code = self.func_mapping[action_name]
            
            if action_code == self.COMPARE_ADJACENT:
                if "index" in params:
                    index = params["index"]
                    msg = self.CompareAdjacentCharacters(index)
                else:
                    msg = "Error: 'index' parameter is missing for COMPARE_ADJACENT action."
            
            elif action_code == self.COUNT_DELETIONS:
                if "comparison_results" in params:
                    comparison_results = params["comparison_results"]
                    msg = self.CountDeletions(comparison_results)
                else:
                    msg = "Error: 'comparison_results' parameter is missing for COUNT_DELETIONS action."
            elif action_code == self.OBSERVE:
                msg = self.Observe()
            elif action_code == self.DONE:
                if "answer" in params:
                    answer = params["answer"]
                    msg = self.Done(answer)
                else:
                    msg = "Error: 'answer' parameter is missing for DONE action."
        except Exception as e:
            msg = f"Error: {str(e)}"

        return True, msg

    # All the actions of the environment
    def CompareAdjacentCharacters(self, index: int):
        r"""
    
        Compare whether the character at the specified index in the string is the same as the next character.
    
        Args:
            index (int): The index of the character to be compared.
    
        Returns:
            str: Returns "True" if the adjacent characters are the same, otherwise returns "False".
    
        Example Output:
            "True"
        """
        if index < 0 or index >= len(self.s) - 1:
            return "Error: index out of range"
        return "True" if self.s[index] == self.s[index + 1] else "False"

    def CountDeletions(self, comparison_results: list):
        r"""
    
        Calculate the number of characters that need to be deleted based on the list of comparison results.
    
        Args:
            comparison_results (list[str]): A list of comparison results, where each element is "True" or "False".
    
        Returns:
            str: The number of characters that need to be deleted.
    
        Example Output:
            "2"
        """
        return str(sum(1 for result in comparison_results if result == "True"))

    def Observe(self):
        r"""
    
        Obtain the string information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "aabcc"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the minimum number of characters that need to be deleted.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = len(s)
        comparison_results = []
        for index in range(n - 1):
            res = self.step(json.dumps({'name': 'CompareAdjacentCharacters', 'parameters': {'index': index}}))[1]
            comparison_results.append(res)
        deletions = self.step(json.dumps({'name': 'CountDeletions', 'parameters': {'comparison_results': comparison_results}}))[1]
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': int(deletions)}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "aabcc"
    env = MinimumDeletionsEnv.from_env_str(f"MinimumDeletionsEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "abcdef"
    env = MinimumDeletionsEnv.from_env_str(f"MinimumDeletionsEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)