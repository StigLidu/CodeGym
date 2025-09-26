# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from collections import Counter

class SmallestWindowEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GET_UNIQUE_CHARS = 0
        self.EXPAND_WINDOW = 1
        self.CONTRACT_WINDOW = 2
        self.CHECK_WINDOW = 3
        self.UPDATE_MIN_SIZE = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "GetUniqueChars": self.GET_UNIQUE_CHARS,
            "ExpandWindow": self.EXPAND_WINDOW,
            "ContractWindow": self.CONTRACT_WINDOW,
            "CheckWindow": self.CHECK_WINDOW,
            "UpdateMinSize": self.UPDATE_MIN_SIZE,
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
        prefix = "SmallestWindowEnv@"
        if not env_str.startswith(prefix):
            return None
        return SmallestWindowEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        self.unique_chars = set(self.s)
        self.unique_count = len(self.unique_chars)
        self.window_counts = Counter()
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        unique_chars = set(self.s)
        unique_count = len(unique_chars)
        left = 0
        right = 0
        min_window_size = float('inf')
        window_counts = Counter()
        formed_unique = 0
        
        while right < self.n:
            char = self.s[right]
            window_counts[char] += 1
            if window_counts[char] == 1:
                formed_unique += 1
                
            while formed_unique == unique_count:
                min_window_size = min(min_window_size, right - left + 1)
                left_char = self.s[left]
                window_counts[left_char] -= 1
                if window_counts[left_char] == 0:
                    formed_unique -= 1
                left += 1
                
            right += 1
            
        return min_window_size

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
            
            if action_code == self.GET_UNIQUE_CHARS:
                msg = self.GetUniqueChars()
            
            elif action_code == self.EXPAND_WINDOW:
                if "right" in params and "current_counts" in params:
                    right = params["right"]
                    current_counts = params["current_counts"]
                    msg = self.ExpandWindow(right, current_counts)
                else:
                    msg = "Error: 'right' or 'current_counts' parameter is missing for EXPAND_WINDOW action."
                    
            elif action_code == self.CONTRACT_WINDOW:
                if "left" in params and "current_counts" in params:
                    left = params["left"]
                    current_counts = params["current_counts"]
                    msg = self.ContractWindow(left, current_counts)
                else:
                    msg = "Error: 'left' or 'current_counts' parameter is missing for CONTRACT_WINDOW action."
                    
            elif action_code == self.CHECK_WINDOW:
                if "current_counts" in params:
                    current_counts = params["current_counts"]
                    msg = self.CheckWindow(current_counts)
                else:
                    msg = "Error: 'current_counts' parameter is missing for CHECK_WINDOW action."
                    
            elif action_code == self.UPDATE_MIN_SIZE:
                if "current_min" in params and "left" in params and "right" in params:
                    current_min = params["current_min"]
                    left = params["left"]
                    right = params["right"]
                    msg = self.UpdateMinSize(current_min, left, right)
                else:
                    msg = "Error: 'current_min', 'left' or 'right' parameter is missing for UPDATE_MIN_SIZE action."
                    
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
    def GetUniqueChars(self):
        r"""
    
        Get all unique characters in the string and their counts.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the list of all unique characters and their counts.
    
        Example Output:
            "{\"unique_chars\": [\"a\", \"b\", \"c\"], \"count\": 3}"
        """
        unique_chars = list(self.unique_chars)
        result = {
            "unique_chars": unique_chars,
            "count": self.unique_count
        }
        return json.dumps(result)

    def ExpandWindow(self, right: int, current_counts: dict):
        r"""
    
        Expand the right boundary of the window and update the character count.
    
        Args:
            right (int): Current position of the right pointer.
            current_counts (dict): Character count of the current window.
    
        Returns:
            str: A JSON string containing the updated right pointer position, character count, and the number of unique characters formed.
    
        Example Output:
            "{\"new_right\": 3, \"new_counts\": {\"a\": 2, \"b\": 1}, \"formed_unique\": 2}"
        """
        if right >= self.n:
            return json.dumps({"error": "Right pointer out of bounds"})
            
        window_counts = Counter(current_counts)
        char = self.s[right]
        window_counts[char] += 1
        
        formed_unique = sum(1 for char in self.unique_chars if window_counts[char] > 0)
        
        result = {
            "new_right": right + 1,
            "new_counts": dict(window_counts),
            "formed_unique": formed_unique
        }
        return json.dumps(result)

    def ContractWindow(self, left: int, current_counts: dict):
        r"""
    
        Shrink the left boundary of the window and update the character count.
    
        Args:
            left (int): Current position of the left pointer.
            current_counts (dict): Character count of the current window.
    
        Returns:
            str: A JSON string containing the updated left pointer position, character count, and the number of unique characters formed.
    
        Example Output:
            "{\"new_left\": 2, \"new_counts\": {\"a\": 1, \"b\": 1}, \"formed_unique\": 2}"
        """
        if left >= self.n:
            return json.dumps({"error": "Left pointer out of bounds"})
            
        window_counts = Counter(current_counts)
        char = self.s[left]
        window_counts[char] -= 1
        
        formed_unique = sum(1 for char in self.unique_chars if window_counts[char] > 0)
        
        result = {
            "new_left": left + 1,
            "new_counts": dict(window_counts),
            "formed_unique": formed_unique
        }
        return json.dumps(result)

    def CheckWindow(self, current_counts: dict):
        r"""
    
        Check if the current window contains all unique characters.
    
        Args:
            current_counts (dict): Character count of the current window.
    
        Returns:
            str: A JSON string containing the check result and the number of unique characters formed.
    
        Example Output:
            "{\"contains_all\": true, \"formed_unique\": 3}"
        """
        window_counts = Counter(current_counts)
        formed_unique = sum(1 for char in self.unique_chars if window_counts[char] > 0)
        contains_all = formed_unique == self.unique_count
        
        result = {
            "contains_all": contains_all,
            "formed_unique": formed_unique
        }
        return json.dumps(result)

    def UpdateMinSize(self, current_min: int, left: int, right: int):
        r"""
    
        Update the minimum window size.
    
        Args:
            current_min (int): Current minimum window size.
            left (int): Current position of the left pointer.
            right (int): Current position of the right pointer.
    
        Returns:
            str: The updated minimum window size.
    
        Example Output:
            "3"
        """
        window_size = right - left
        new_min = min(current_min, window_size)
        return str(new_min)

    def Observe(self):
        r"""
    
        Return the currently observable environmental information.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the length of the string and the string itself.
    
        Example Output:
            "{\"n\": 7, \"s\": \"aabcabb\"}"
        """
        result = {
            "n": self.n,
            "s": self.s
        }
        return json.dumps(result)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the length of the minimum window.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 3, Reference answer: 3, Result: Correct, reward=1"
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
        import json
    
        observe_result = self.step(json.dumps({"name": "Observe", "parameters": {}}))[1]
        observe_data = json.loads(observe_result)
        n = observe_data["n"]
        s = observe_data["s"]
    
        unique_chars_result = self.step(json.dumps({"name": "GetUniqueChars", "parameters": {}}))[1]
        unique_data = json.loads(unique_chars_result)
        required_chars_count = unique_data["count"]
    
        left = 0
        right = 0
        current_counts = {}
        current_min = float("inf")
        formed_unique = 0
    
        while right < n:
            expand_result = self.step(json.dumps({
                "name": "ExpandWindow",
                "parameters": {"right": right, "current_counts": current_counts}
            }))[1]
            expand_data = json.loads(expand_result)
            right = expand_data["new_right"]
            current_counts = expand_data["new_counts"]
            formed_unique = expand_data["formed_unique"]
    
            while formed_unique == required_chars_count and left <= right:
                current_min = int(self.step(json.dumps({
                    "name": "UpdateMinSize",
                    "parameters": {"current_min": current_min, "left": left, "right": right}
                }))[1])
    
                contract_result = self.step(json.dumps({
                    "name": "ContractWindow",
                    "parameters": {"left": left, "current_counts": current_counts}
                }))[1]
                contract_data = json.loads(contract_result)
                left = contract_data["new_left"]
                current_counts = contract_data["new_counts"]
                formed_unique = contract_data["formed_unique"]
    
        return self.step(json.dumps({"name": "Done", "parameters": {"answer": current_min}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_n1 = 7
    test_s1 = "aabcabb"
    env1 = SmallestWindowEnv.from_env_str(f"SmallestWindowEnv@{{\"n\": {test_n1}, \"s\": \"{test_s1}\"}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_n2 = 5
    test_s2 = "abcde"
    env2 = SmallestWindowEnv.from_env_str(f"SmallestWindowEnv@{{\"n\": {test_n2}, \"s\": \"{test_s2}\"}}")
    print(env2.solve())
    print("step count:", env2.step_count)