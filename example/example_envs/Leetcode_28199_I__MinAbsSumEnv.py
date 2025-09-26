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

class MinAbsSumEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.SORT_ARRAY = 0
        self.CALCULATE_ABS_DIFF = 1
        self.SUM_NUMBERS = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "SortArray": self.SORT_ARRAY,
            "CalculateAbsDiff": self.CALCULATE_ABS_DIFF,
            "SumNumbers": self.SUM_NUMBERS,
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
        prefix = "MinAbsSumEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinAbsSumEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.arr1 = options.get("arr1", [])
        self.arr2 = options.get("arr2", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        sorted_arr1 = sorted(self.arr1)
        sorted_arr2 = sorted(self.arr2)
        return sum(abs(a - b) for a, b in zip(sorted_arr1, sorted_arr2))

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
            
            if action_code == self.SORT_ARRAY:
                if "array" in params:
                    array = params["array"]
                    msg = self.SortArray(array)
                else:
                    msg = "Error: 'array' parameter is missing for SORT_ARRAY action."
            
            elif action_code == self.CALCULATE_ABS_DIFF:
                if "a" in params and "b" in params:
                    a = params["a"]
                    b = params["b"]
                    msg = self.CalculateAbsDiff(a, b)
                else:
                    msg = "Error: 'a' or 'b' parameter is missing for CALCULATE_ABS_DIFF action."
                    
            elif action_code == self.SUM_NUMBERS:
                if "numbers" in params:
                    numbers = params["numbers"]
                    msg = self.SumNumbers(numbers)
                else:
                    msg = "Error: 'numbers' parameter is missing for SUM_NUMBERS action."
                    
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
    def SortArray(self, array: list):
        r"""
    
        Sort the input array in ascending order.
    
        Args:
            array (list[int]): The integer array to be sorted.
    
        Returns:
            str: The sorted array returned as a JSON string.
    
        Example Output:
            "[1, 2, 3, 4, 5]"
        """
        sorted_array = sorted(array)
        return json.dumps(sorted_array)

    def CalculateAbsDiff(self, a: int, b: int):
        r"""
    
        Calculate the absolute difference between two integers.
    
        Args:
            a (int): The first integer.
            b (int): The second integer.
    
        Returns:
            str: The absolute difference between the two integers.
    
        Example Output:
            "3"
        """
        return str(abs(a - b))

    def SumNumbers(self, numbers: list):
        r"""
    
        Calculate the sum of a set of numbers.
    
        Args:
            numbers (list[int]): The list of numbers to be summed.
    
        Returns:
            str: The sum of the number list.
    
        Example Output:
            "15"
        """
        return str(sum(numbers))

    def Observe(self):
        r"""
    
        Obtain two arrays in the current environment.
    
        Args:
            None
    
        Returns:
            str: A prompt containing information about the two arrays.
    
        Example Output:
            "arr1: [3, 1, 2], arr2: [2, 3, 1]"
        """
        return f"arr1: {self.arr1}, arr2: {self.arr2}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The sum of the minimum absolute differences submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 0, Reference answer: 0, Result: Correct, reward=1"
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
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        import ast
        arr1_str = observe_result.split('arr1: ')[1].split(', arr2: ')[0]
        arr2_str = observe_result.split('arr2: ')[1]
        arr1 = ast.literal_eval(arr1_str)
        arr2 = ast.literal_eval(arr2_str)
        
        sorted_arr1_str = self.step(json.dumps({'name': 'SortArray', 'parameters': {'array': arr1}}))[1]
        sorted_arr1 = ast.literal_eval(sorted_arr1_str)
        
        sorted_arr2_str = self.step(json.dumps({'name': 'SortArray', 'parameters': {'array': arr2}}))[1]
        sorted_arr2 = ast.literal_eval(sorted_arr2_str)
        
        diffs = []
        for a, b in zip(sorted_arr1, sorted_arr2):
            diff_str = self.step(json.dumps({'name': 'CalculateAbsDiff', 'parameters': {'a': a, 'b': b}}))[1]
            diffs.append(int(diff_str))
        
        total_diff_str = self.step(json.dumps({'name': 'SumNumbers', 'parameters': {'numbers': diffs}}))[1]
        total_diff = int(total_diff_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total_diff}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_arr1 = [3, 1, 2]
    test_arr2 = [2, 3, 1]
    env = MinAbsSumEnv.from_env_str(f"MinAbsSumEnv@{{\"arr1\": {test_arr1}, \"arr2\": {test_arr2}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_arr1 = [random.randint(1, 100) for _ in range(5)]
    test_arr2 = [random.randint(1, 100) for _ in range(5)]
    env = MinAbsSumEnv.from_env_str(f"MinAbsSumEnv@{{\"arr1\": {test_arr1}, \"arr2\": {test_arr2}}}")
    print(env.solve())
    print("step count:", env.step_count)