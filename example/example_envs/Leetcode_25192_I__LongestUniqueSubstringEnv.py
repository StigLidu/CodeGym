# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestUniqueSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.UPDATE_CHAR_MAP = 1
        self.UPDATE_LEFT_POINTER = 2
        self.CALCULATE_CURRENT_LENGTH = 3
        self.UPDATE_MAX_LENGTH = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "UpdateCharMap": self.UPDATE_CHAR_MAP,
            "UpdateLeftPointer": self.UPDATE_LEFT_POINTER,
            "CalculateCurrentLength": self.CALCULATE_CURRENT_LENGTH,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
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
        prefix = "LongestUniqueSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestUniqueSubstringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        self.char_map = {}
        self.left = 0
        self.max_length = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(self.s)):
            if self.s[right] in char_map and char_map[self.s[right]] >= left:
                left = char_map[self.s[right]] + 1
            char_map[self.s[right]] = right
            max_length = max(max_length, right - left + 1)
            
        return max_length

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
            
            if action_code == self.OBSERVE:
                msg = self.Observe()
            
            elif action_code == self.UPDATE_CHAR_MAP:
                if "char" in params and "position" in params:
                    char = params["char"]
                    position = params["position"]
                    msg = self.UpdateCharMap(char, position)
                else:
                    msg = "Error: 'char' or 'position' parameter is missing for UPDATE_CHAR_MAP action."
                    
            elif action_code == self.UPDATE_LEFT_POINTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.UpdateLeftPointer(char)
                else:
                    msg = "Error: 'char' parameter is missing for UPDATE_LEFT_POINTER action."
                    
            elif action_code == self.CALCULATE_CURRENT_LENGTH:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateCurrentLength(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_CURRENT_LENGTH action."
                    
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "current_length" in params:
                    current_length = params["current_length"]
                    msg = self.UpdateMaxLength(current_length)
                else:
                    msg = "Error: 'current_length' parameter is missing for UPDATE_MAX_LENGTH action."
                    
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
    def Observe(self):
        r"""
    
        Obtain the input string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The input string in the current environment.
    
        Example Output:
            "abcabcbb"
        """
        return self.s

    def UpdateCharMap(self, char: str, position: int):
        r"""
    
        Update the character position mapping table to record the most recent occurrence position of the character.
    
        Args:
            char (str): The character to be recorded.
            position (int): The position where the character appears.
    
        Returns:
            str: The updated character position mapping table.
    
        Example Output:
            "{'a': 0, 'b': 1, 'c': 2}"
        """
        self.char_map[char] = position
        return json.dumps(self.char_map)

    def UpdateLeftPointer(self, char: str):
        r"""
    
        Update the left pointer based on the position of the character in the mapping table.
    
        Args:
            char (str): The character to be checked.
    
        Returns:
            str: The updated position of the left pointer.
    
        Example Output:
            "3"
        """
        if char in self.char_map and self.char_map[char] >= self.left:
            self.left = self.char_map[char] + 1
        return str(self.left)

    def CalculateCurrentLength(self, left: int, right: int):
        r"""
    
        Calculate the length of the current window.
    
        Args:
            left (int): The position of the left pointer.
            right (int): The position of the right pointer.
    
        Returns:
            str: The length of the current window.
    
        Example Output:
            "3"
        """
        current_length = right - left + 1
        return str(current_length)

    def UpdateMaxLength(self, current_length: int):
        r"""
    
        Update the maximum window length.
    
        Args:
            current_length (int): The current window length.
    
        Returns:
            str: The updated maximum window length.
    
        Example Output:
            "3"
        """
        self.max_length = max(self.max_length, current_length)
        return str(self.max_length)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
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
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        left = 0
        max_length = 0
        for right in range(len(s)):
            current_char = s[right]
            new_left = int(self.step(json.dumps({'name': 'UpdateLeftPointer', 'parameters': {'char': current_char}}))[1])
            if new_left > left:
                left = new_left
            current_length = int(self.step(json.dumps({'name': 'CalculateCurrentLength', 'parameters': {'left': left, 'right': right}}))[1])
            max_length = int(self.step(json.dumps({'name': 'UpdateMaxLength', 'parameters': {'current_length': current_length}}))[1])
            self.step(json.dumps({'name': 'UpdateCharMap', 'parameters': {'char': current_char, 'position': right}}))
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "abcabcbb"
    env = LongestUniqueSubstringEnv.from_env_str(f"LongestUniqueSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "bbbbb"
    env = LongestUniqueSubstringEnv.from_env_str(f"LongestUniqueSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)