# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MaxConsecutivePairsEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_ADJACENT = 0
        self.UPDATE_COUNTERS = 1
        self.GET_MAX_PAIRS = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckAdjacent": self.CHECK_ADJACENT,
            "UpdateCounters": self.UPDATE_COUNTERS,
            "GetMaxPairs": self.GET_MAX_PAIRS,
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
        prefix = "MaxConsecutivePairsEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxConsecutivePairsEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.gem_string = options.get("gem_string", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        max_pairs = 0
        current_pairs = 0
        n = len(self.gem_string)
        
        for i in range(1, n):
            if self.gem_string[i] == self.gem_string[i - 1]:
                current_pairs += 1
                max_pairs = max(max_pairs, current_pairs)
            else:
                current_pairs = 0
        
        return max_pairs

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
            
            if action_code == self.CHECK_ADJACENT:
                if "index" in params:
                    index = params["index"]
                    msg = self.CheckAdjacent(index)
                else:
                    msg = "Error: 'index' parameter is missing for CHECK_ADJACENT action."
            
            elif action_code == self.UPDATE_COUNTERS:
                if "is_same" in params and "current_pairs" in params and "max_pairs" in params:
                    is_same = params["is_same"]
                    current_pairs = params["current_pairs"]
                    max_pairs = params["max_pairs"]
                    msg = self.UpdateCounters(is_same, current_pairs, max_pairs)
                else:
                    msg = "Error: 'is_same', 'current_pairs' or 'max_pairs' parameter is missing for UPDATE_COUNTERS action."
                    
            elif action_code == self.GET_MAX_PAIRS:
                if "max_pairs" in params:
                    max_pairs = params["max_pairs"]
                    msg = self.GetMaxPairs(max_pairs)
                else:
                    msg = "Error: 'max_pairs' parameter is missing for GET_MAX_PAIRS action."
                    
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
    def CheckAdjacent(self, index: int):
        r"""
    
        Check if the gem at the specified index is the same as the gem at the previous index.
    
        Args:
            index (int): The current index position to check.
    
        Returns:
            str: "True" indicates they are the same, "False" indicates they are different.
    
        Example Output:
            "True"
        """
        if 1 <= index < len(self.gem_string):
            return str(self.gem_string[index] == self.gem_string[index - 1])
        return "False"

    def UpdateCounters(self, is_same: bool, current_pairs: int, max_pairs: int):
        r"""
    
        Update the current consecutive pair count and the maximum consecutive pair count based on whether adjacent gems are the same.
    
        Args:
            is_same (bool): Whether adjacent gems are the same.
            current_pairs (int): Current consecutive pair count.
            max_pairs (int): Current maximum consecutive pair count.
    
        Returns:
            str: The updated current consecutive pair count and maximum consecutive pair count, formatted as "current,max".
    
        Example Output:
            "2,2"
        """
        if is_same:
            current_pairs += 1
            max_pairs = max(max_pairs, current_pairs)
        else:
            current_pairs = 0
        return f"{current_pairs},{max_pairs}"

    def GetMaxPairs(self, max_pairs: int):
        r"""
    
        Return the maximum consecutive pair count.
    
        Args:
            max_pairs (int): Current maximum consecutive pair count.
    
        Returns:
            str: The maximum consecutive pair count.
    
        Example Output:
            "2"
        """
        return str(max_pairs)

    def Observe(self):
        r"""
    
        Return the information of the current gem sequence.
    
        Args:
            None
    
        Returns:
            str: The current gem sequence.
    
        Example Output:
            "aabbbcc"
        """
        return self.gem_string

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
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
        gem_sequence = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        sequence_length = len(gem_sequence)
        
        current_pairs = 0
        max_pairs = 0
        
        for index in range(1, sequence_length):
            is_same_str = self.step(json.dumps({'name': 'CheckAdjacent', 'parameters': {'index': index}}))[1]
            is_same = is_same_str == "True"
            
            update_result = self.step(json.dumps({
                'name': 'UpdateCounters',
                'parameters': {
                    'is_same': is_same,
                    'current_pairs': current_pairs,
                    'max_pairs': max_pairs
                }
            }))[1]
            current_pairs, max_pairs = map(int, update_result.split(','))
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_pairs}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_gems1 = "aabbbcc"
    env1 = MaxConsecutivePairsEnv.from_env_str(f"MaxConsecutivePairsEnv@{{\"gem_string\": \"{test_gems1}\"}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_gems2 = "abcabcabcd"
    env2 = MaxConsecutivePairsEnv.from_env_str(f"MaxConsecutivePairsEnv@{{\"gem_string\": \"{test_gems2}\"}}")
    print(env2.solve())
    print("step count:", env2.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_gems3 = "aabaa"
    env3 = MaxConsecutivePairsEnv.from_env_str(f"MaxConsecutivePairsEnv@{{\"gem_string\": \"{test_gems3}\"}}")
    print(env3.solve())
    print("step count:", env3.step_count)