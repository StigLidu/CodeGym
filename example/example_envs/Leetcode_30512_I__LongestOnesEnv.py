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
        self.FIND_ZERO_POSITIONS = 0
        self.EXPAND_LEFT = 1
        self.EXPAND_RIGHT = 2
        self.CALCULATE_LENGTH = 3
        self.FIND_MAX_LENGTH = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "FindZeroPositions": self.FIND_ZERO_POSITIONS,
            "ExpandLeft": self.EXPAND_LEFT,
            "ExpandRight": self.EXPAND_RIGHT,
            "CalculateLength": self.CALCULATE_LENGTH,
            "FindMaxLength": self.FIND_MAX_LENGTH,
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
        prefix = "LongestOnesEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestOnesEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.binary_str = options.get("binary_str", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        max_len = 0
        n = len(self.binary_str)
        zeros_positions = [i for i, char in enumerate(self.binary_str) if char == '0']
        
        if not zeros_positions:         # No '0' in the string
            return n
        
        for pos in zeros_positions:
            left = right = pos
            
            # Expand left
            while left > 0 and self.binary_str[left - 1] == '1':
                left -= 1
            
            # Expand right
            while right < n - 1 and self.binary_str[right + 1] == '1':
                right += 1
            
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
            
            if action_code == self.FIND_ZERO_POSITIONS:
                msg = self.FindZeroPositions()
            
            elif action_code == self.EXPAND_LEFT:
                if "position" in params:
                    position = params["position"]
                    msg = self.ExpandLeft(position)
                else:
                    msg = "Error: 'position' parameter is missing for EXPAND_LEFT action."
            
            elif action_code == self.EXPAND_RIGHT:
                if "position" in params:
                    position = params["position"]
                    msg = self.ExpandRight(position)
                else:
                    msg = "Error: 'position' parameter is missing for EXPAND_RIGHT action."
                    
            elif action_code == self.CALCULATE_LENGTH:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateLength(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_LENGTH action."
                    
            elif action_code == self.FIND_MAX_LENGTH:
                if "lengths" in params:
                    lengths = params["lengths"]
                    msg = self.FindMaxLength(lengths)
                else:
                    msg = "Error: 'lengths' parameter is missing for FIND_MAX_LENGTH action."
                    
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
    def FindZeroPositions(self):
        r"""
    
        Find all positions of '0's in the binary string.
    
        Args:
            None
    
        Returns:
            str: A list string containing all positions of '0's.
    
        Example Output:
            "[1, 3, 5]"
        """
        positions = [i for i, char in enumerate(self.binary_str) if char == '0']
        return str(positions)

    def ExpandLeft(self, position: int):
        r"""
    
        Expand leftward from the specified position to find the leftmost boundary of consecutive '1's.
    
        Args:
            position (int): Starting position.
    
        Returns:
            str: The left boundary position after expansion.
    
        Example Output:
            "0"
        """
        left = position
        while left > 0 and self.binary_str[left - 1] == '1':
            left -= 1
        return str(left)

    def ExpandRight(self, position: int):
        r"""
    
        Expand rightward from the specified position to find the rightmost boundary of consecutive '1's.
    
        Args:
            position (int): Starting position.
    
        Returns:
            str: The right boundary position after expansion.
    
        Example Output:
            "5"
        """
        right = position
        n = len(self.binary_str)
        while right < n - 1 and self.binary_str[right + 1] == '1':
            right += 1
        return str(right)

    def CalculateLength(self, left: int, right: int):
        r"""
    
        Calculate the length of the sequence from left to right.
    
        Args:
            left (int): Left boundary position.
            right (int): Right boundary position.
    
        Returns:
            str: Length of the sequence.
    
        Example Output:
            "6"
        """
        return str(right - left + 1)

    def FindMaxLength(self, lengths: list):
        r"""
    
        Find the maximum value in the length list.
    
        Args:
            lengths (list[int]): List of lengths.
    
        Returns:
            str: Maximum length.
    
        Example Output:
            "6"
        """
        return str(max(lengths))

    def Observe(self):
        r"""
    
        Return the current binary string.
    
        Args:
            None
    
        Returns:
            str: The current binary string.
    
        Example Output:
            "110111"
        """
        return self.binary_str

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 6, Reference answer: 6, Result: Correct, reward=1"
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
        binary_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        zero_positions_str = self.step(json.dumps({'name': 'FindZeroPositions', 'parameters': {}}))[1]
        zero_positions = ast.literal_eval(zero_positions_str)
        
        lengths = []
        if zero_positions:
            for pos in zero_positions:
                left = int(self.step(json.dumps({'name': 'ExpandLeft', 'parameters': {'position': pos}}))[1])
                right = int(self.step(json.dumps({'name': 'ExpandRight', 'parameters': {'position': pos}}))[1])
                length = int(self.step(json.dumps({'name': 'CalculateLength', 'parameters': {'left': left, 'right': right}}))[1])
                lengths.append(length)
            max_len = int(self.step(json.dumps({'name': 'FindMaxLength', 'parameters': {'lengths': lengths}}))[1])
        else:
            max_len = len(binary_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_len}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "110111"
    env = LongestOnesEnv.from_env_str(f"LongestOnesEnv@{{\"binary_str\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "1010101"
    env = LongestOnesEnv.from_env_str(f"LongestOnesEnv@{{\"binary_str\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)