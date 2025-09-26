# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
import random

class LongestEvenSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_EVEN = 1
        self.UPDATE_COUNTERS = 2
        self.FINAL_RESULT = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckEven": self.CHECK_EVEN,
            "UpdateCounters": self.UPDATE_COUNTERS,
            "FinalResult": self.FINAL_RESULT,
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
        prefix = "LongestEvenSubsequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestEvenSubsequenceEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.sequence = options.get("sequence", [])
        self.max_length = 0
        self.current_length = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        max_length = 0
        current_length = 0
        
        for num in self.sequence:
            if num % 2 == 0:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length if max_length > 0 else -1

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
            
            elif action_code == self.CHECK_EVEN:
                if "number" in params:
                    number = params["number"]
                    msg = self.CheckEven(number)
                else:
                    msg = "Error: 'number' parameter is missing for CHECK_EVEN action."
            
            elif action_code == self.UPDATE_COUNTERS:
                if "is_even" in params:
                    is_even = params["is_even"]
                    msg = self.UpdateCounters(is_even)
                else:
                    msg = "Error: 'is_even' parameter is missing for UPDATE_COUNTERS action."
            
            elif action_code == self.FINAL_RESULT:
                msg = self.FinalResult()
            
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
    
        Obtain the current integer sequence.
    
        Args:
            None
    
        Returns:
            str: The current integer sequence, represented as a JSON format string.
    
        Example Output:
            "[1, 2, 4, 6, 8, 3, 5, 7]"
        """
        return json.dumps(self.sequence)

    def CheckEven(self, number: int):
        r"""
    
        Check if a number is even.
    
        Args:
            number (int): The number to be checked.
    
        Returns:
            str: "true" indicates it is an even number, "false" indicates it is not an even number.
    
        Example Output:
            "true"
        """
        return "true" if number % 2 == 0 else "false"

    def UpdateCounters(self, is_even: bool):
        r"""
    
        Update the current consecutive even length and the maximum consecutive even length based on whether the number is even.
    
        Args:
            is_even (bool): Whether the current number is even.
    
        Returns:
            str: The updated current length and maximum length, in the format "current_length,max_length".
    
        Example Output:
            "3,4"
        """
        if is_even:
            self.current_length += 1
            if self.current_length > self.max_length:
                self.max_length = self.current_length
        else:
            self.current_length = 0
        return f"{self.current_length},{self.max_length}"

    def FinalResult(self):
        r"""
    
        Calculate the final result based on the maximum consecutive even length. If there are no even numbers, return -1.
    
        Args:
            None
    
        Returns:
            str: The final result, which is the length of the longest consecutive even subsequence or -1.
    
        Example Output:
            "4"
        """
        return str(self.max_length if self.max_length > 0 else -1)

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 4, Reference answer: 4, Result: Correct, reward=1"
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
        sequence_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        sequence = json.loads(sequence_str)
        
        for num in sequence:
            is_even_str = self.step(json.dumps({'name': 'CheckEven', 'parameters': {'number': num}}))[1]
            is_even = (is_even_str == "true")
            self.step(json.dumps({'name': 'UpdateCounters', 'parameters': {'is_even': is_even}}))
        
        final_result = int(self.step(json.dumps({'name': 'FinalResult', 'parameters': {}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_sequence1 = [1, 2, 4, 6, 8, 3, 5, 7]
    env1 = LongestEvenSubsequenceEnv.from_env_str(f"LongestEvenSubsequenceEnv@{{\"sequence\": {test_sequence1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)
    print()
    
    # test case 2
    print("Test Case 2:")
    test_sequence2 = [5, 3, 9, 1, 7]
    env2 = LongestEvenSubsequenceEnv.from_env_str(f"LongestEvenSubsequenceEnv@{{\"sequence\": {test_sequence2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)
    print()
    
    # test case 3 (additional test case)
    print("Test Case 3:")
    test_sequence3 = [2, 4, 6, 8, 10, 12]
    env3 = LongestEvenSubsequenceEnv.from_env_str(f"LongestEvenSubsequenceEnv@{{\"sequence\": {test_sequence3}}}")
    print(env3.solve())
    print("step count:", env3.step_count)