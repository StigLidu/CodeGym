# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import math
import ast
import json
from functools import reduce

class MaxGCDAfterDivideEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.DIVIDE_ELEMENT_BY_INDEX = 1
        self.CALCULATE_GCD = 2
        self.SET_CANDIDATE_GCD = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "DivideElementByIndex": self.DIVIDE_ELEMENT_BY_INDEX,
            "CalculateGCD": self.CALCULATE_GCD,
            "SetCandidateGCD": self.SET_CANDIDATE_GCD,
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
        prefix = "MaxGCDAfterDivideEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxGCDAfterDivideEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.original_array = options.get("array", [])
        self.current_array = self.original_array.copy()
        self.candidate_gcd = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        max_possible_gcd = 0
        for i in range(len(self.original_array)):
            modified_arr = self.original_array.copy()
            modified_arr[i] //= 2
            modified_gcd = reduce(math.gcd, modified_arr)
            max_possible_gcd = max(max_possible_gcd, modified_gcd)
        return max_possible_gcd

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
            
            elif action_code == self.DIVIDE_ELEMENT_BY_INDEX:
                if "index" in params:
                    index = params["index"]
                    msg = self.DivideElementByIndex(index)
                else:
                    msg = "Error: 'index' parameter is missing for DIVIDE_ELEMENT_BY_INDEX action."
            
            elif action_code == self.CALCULATE_GCD:
                if "array" in params:
                    array = params["array"]
                    msg = self.CalculateGCD(array)
                else:
                    msg = "Error: 'array' parameter is missing for CALCULATE_GCD action."
            
            elif action_code == self.SET_CANDIDATE_GCD:
                if "value" in params:
                    value = params["value"]
                    msg = self.SetCandidateGCD(value)
                else:
                    msg = "Error: 'value' parameter is missing for SET_CANDIDATE_GCD action."
            
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
    
        Obtain information about the current array.
    
        Args:
            None
    
        Returns:
            str: String representation of the current array.
    
        Example Output:
            "[8, 4, 6]"
        """
        return str(self.original_array)

    def DivideElementByIndex(self, index: int):
        r"""
    
        Divide the element at the specified index in the array by 2, and return the modified array.
    
        Args:
            index (int): The index of the element to be modified.
    
        Returns:
            str: String representation of the modified array.
    
        Example Output:
            "[4, 4, 6]"
        """
        if 0 <= index < len(self.original_array):
            modified_array = self.original_array.copy()
            modified_array[index] //= 2
            return str(modified_array)
        else:
            return "Error: Invalid index"

    def CalculateGCD(self, array: list):
        r"""
    
        Calculate the GCD of the given array.
    
        Args:
            array (list[int]): The array for which to calculate the GCD.
    
        Returns:
            str: The GCD value of the array.
    
        Example Output:
            "2"
        """
        if not array:
            return "Error: Empty array"
        current_gcd = array[0]
        for num in array[1:]:
            current_gcd = math.gcd(current_gcd, num)
            if current_gcd == 1:
                break  # GCD can't be smaller than 1
        return str(current_gcd)

    def SetCandidateGCD(self, value: int):
        r"""
    
        Set the candidate maximum GCD value; update if the new value is larger.
    
        Args:
            value (int): The new candidate GCD value.
    
        Returns:
            str: The updated candidate GCD value.
    
        Example Output:
            "4"
        """
        if value > self.candidate_gcd:
            self.candidate_gcd = value
        return str(self.candidate_gcd)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
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
        import ast
        initial_array_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        initial_array = ast.literal_eval(initial_array_str)
        n = len(initial_array)
        
        self.step(json.dumps({'name': 'SetCandidateGCD', 'parameters': {'value': 0}}))
        
        for i in range(n):
            modified_array_str = self.step(json.dumps({'name': 'DivideElementByIndex', 'parameters': {'index': i}}))[1]
            modified_array = ast.literal_eval(modified_array_str)
            current_gcd_str = self.step(json.dumps({'name': 'CalculateGCD', 'parameters': {'array': modified_array}}))[1]
            current_gcd = int(current_gcd_str)
            self.step(json.dumps({'name': 'SetCandidateGCD', 'parameters': {'value': current_gcd}}))
        
        
        max_gcd = 0
        for i in range(n):
            modified_array_str = self.step(json.dumps({'name': 'DivideElementByIndex', 'parameters': {'index': i}}))[1]
            modified_array = ast.literal_eval(modified_array_str)
            current_gcd_str = self.step(json.dumps({'name': 'CalculateGCD', 'parameters': {'array': modified_array}}))[1]
            current_gcd = int(current_gcd_str)
            if current_gcd > max_gcd:
                max_gcd = current_gcd
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_gcd}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (from example)
    print("Test Case 1:")
    test_array = [8, 4, 6]
    env = MaxGCDAfterDivideEnv.from_env_str(f"MaxGCDAfterDivideEnv@{{\"array\": {test_array}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2 (from example)
    print("\nTest Case 2:")
    test_array = [15, 30, 45, 60]
    env = MaxGCDAfterDivideEnv.from_env_str(f"MaxGCDAfterDivideEnv@{{\"array\": {test_array}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 3 (from example)
    print("\nTest Case 3:")
    test_array = [10, 20]
    env = MaxGCDAfterDivideEnv.from_env_str(f"MaxGCDAfterDivideEnv@{{\"array\": {test_array}}}")
    print(env.solve())
    print("step count:", env.step_count)