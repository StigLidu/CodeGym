# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestOnesEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_STRING_LENGTH = 1
        self.CHECK_ZERO_EXISTENCE = 2
        self.COUNT_ZEROS_IN_WINDOW = 3
        self.CALCULATE_WINDOW_LENGTH = 4
        self.UPDATE_MAX_LENGTH = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "CheckZeroExistence": self.CHECK_ZERO_EXISTENCE,
            "CountZerosInWindow": self.COUNT_ZEROS_IN_WINDOW,
            "CalculateWindowLength": self.CALCULATE_WINDOW_LENGTH,
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
        prefix = "LongestOnesEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestOnesEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.s)
        if '0' not in self.s:
            return n
        
        max_len = 0
        left = 0
        count_zeros = 0
        
        for right in range(n):
            if self.s[right] == '0':
                count_zeros += 1
            
            while count_zeros > 1:
                if self.s[left] == '0':
                    count_zeros -= 1
                left += 1

            max_len = max(max_len, right - left + 1)
        
        return max_len

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
                
            elif action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
                
            elif action_code == self.CHECK_ZERO_EXISTENCE:
                msg = self.CheckZeroExistence()
                
            elif action_code == self.COUNT_ZEROS_IN_WINDOW:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CountZerosInWindow(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for COUNT_ZEROS_IN_WINDOW action."
                    
            elif action_code == self.CALCULATE_WINDOW_LENGTH:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateWindowLength(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_WINDOW_LENGTH action."
                    
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "current_max" in params and "new_value" in params:
                    current_max = params["current_max"]
                    new_value = params["new_value"]
                    msg = self.UpdateMaxLength(current_max, new_value)
                else:
                    msg = "Error: 'current_max' or 'new_value' parameter is missing for UPDATE_MAX_LENGTH action."
                    
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
    
        Returns the binary string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The binary string in the current environment.
    
        Example Output:
            "1101100111"
        """
        return self.s

    def GetStringLength(self):
        r"""
    
        Returns the length of the binary string.
    
        Args:
            None
    
        Returns:
            str: The length of the binary string.
    
        Example Output:
            "10"
        """
        return str(len(self.s))

    def CheckZeroExistence(self):
        r"""
    
        Checks whether the character '0' exists in the binary string.
    
        Args:
            None
    
        Returns:
            str: Returns "True" if '0' exists, otherwise returns "False".
    
        Example Output:
            "True"
        """
        return str('0' in self.s)

    def CountZerosInWindow(self, left: int, right: int):
        r"""
    
        Counts the number of '0's within the specified window range.
    
        Args:
            left (int): The left boundary index of the window.
            right (int): The right boundary index of the window.
    
        Returns:
            str: The number of '0's in the window.
    
        Example Output:
            "1"
        """
        count = 0
        for i in range(left, right + 1):
            if self.s[i] == '0':
                count += 1
        return str(count)

    def CalculateWindowLength(self, left: int, right: int):
        r"""
    
        Calculates the length of the specified window.
    
        Args:
            left (int): The left boundary index of the window.
            right (int): The right boundary index of the window.
    
        Returns:
            str: The length of the window.
    
        Example Output:
            "5"
        """
        return str(right - left + 1)

    def UpdateMaxLength(self, current_max: int, new_value: int):
        r"""
    
        Compares the current maximum value with the new value and returns the larger one.
    
        Args:
            current_max (int): The current maximum value.
            new_value (int): The new candidate value.
    
        Returns:
            str: The updated maximum value.
    
        Example Output:
            "5"
        """
        return str(max(current_max, new_value))

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 5, Reference answer: 5, Result: Correct, reward=1"
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
        has_zero = self.step(json.dumps({'name': 'CheckZeroExistence', 'parameters': {}}))[1]
        if has_zero == "False":
            str_len = int(self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1])
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': str_len}}))[1]
        
        str_len = int(self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1])
        left = 0
        max_length = 0
        
        for right in range(str_len):
            zero_count = int(self.step(json.dumps({
                'name': 'CountZerosInWindow', 
                'parameters': {'left': left, 'right': right}
            }))[1])
            
            while zero_count > 1:
                left += 1
                zero_count = int(self.step(json.dumps({
                    'name': 'CountZerosInWindow', 
                    'parameters': {'left': left, 'right': right}
                }))[1])
            
            current_window_len = int(self.step(json.dumps({
                'name': 'CalculateWindowLength', 
                'parameters': {'left': left, 'right': right}
            }))[1])
            
            max_length = int(self.step(json.dumps({
                'name': 'UpdateMaxLength', 
                'parameters': {'current_max': max_length, 'new_value': current_window_len}
            }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "1101100111"
    env = LongestOnesEnv.from_env_str(f"LongestOnesEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "1111"
    env = LongestOnesEnv.from_env_str(f"LongestOnesEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)