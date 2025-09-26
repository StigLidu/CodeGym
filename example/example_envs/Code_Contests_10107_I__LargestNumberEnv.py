# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from functools import cmp_to_key

class LargestNumberEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CONVERT_TO_STRINGS = 0
        self.COMPARE_TWO_ELEMENTS = 1
        self.SORT_LIST = 2
        self.CONCATENATE_LIST = 3
        self.CHECK_LEADING_ZERO = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "ConvertToStrings": self.CONVERT_TO_STRINGS,
            "CompareTwoElements": self.COMPARE_TWO_ELEMENTS,
            "SortList": self.SORT_LIST,
            "ConcatenateList": self.CONCATENATE_LIST,
            "CheckLeadingZero": self.CHECK_LEADING_ZERO,
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
        prefix = "LargestNumberEnv@"
        if not env_str.startswith(prefix):
            return None
        return LargestNumberEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.arr = options.get("arr", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def compare(x, y):
            if x + y > y + x:
                return -1
            elif x + y < y + x:
                return 1
            else:
                return 0
                
        arr = list(map(str, self.arr))
        arr.sort(key=cmp_to_key(compare))
        largest_num = ''.join(arr)
        
        if largest_num[0] == '0':
            return '0'
        
        return largest_num

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
            
            if action_code == self.CONVERT_TO_STRINGS:
                if "numbers" in params:
                    numbers = params["numbers"]
                    msg = self.ConvertToStrings(numbers)
                else:
                    msg = "Error: 'numbers' parameter is missing for CONVERT_TO_STRINGS action."
            
            elif action_code == self.COMPARE_TWO_ELEMENTS:
                if "x" in params and "y" in params:
                    x = params["x"]
                    y = params["y"]
                    msg = self.CompareTwoElements(x, y)
                else:
                    msg = "Error: 'x' or 'y' parameter is missing for COMPARE_TWO_ELEMENTS action."
                    
            elif action_code == self.SORT_LIST:
                if "arr" in params and "comparison_results" in params:
                    arr = params["arr"]
                    comparison_results = params["comparison_results"]
                    msg = self.SortList(arr, comparison_results)
                else:
                    msg = "Error: 'arr' or 'comparison_results' parameter is missing for SORT_LIST action."
                    
            elif action_code == self.CONCATENATE_LIST:
                if "arr" in params:
                    arr = params["arr"]
                    msg = self.ConcatenateList(arr)
                else:
                    msg = "Error: 'arr' parameter is missing for CONCATENATE_LIST action."
                    
            elif action_code == self.CHECK_LEADING_ZERO:
                if "s" in params:
                    s = params["s"]
                    msg = self.CheckLeadingZero(s)
                else:
                    msg = "Error: 's' parameter is missing for CHECK_LEADING_ZERO action."
                    
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
    def ConvertToStrings(self, numbers: list):
        r"""
    
        Convert a list of integers into a list of strings.
    
        Args:
            numbers (list[int]): The list of integers to be converted.
    
        Returns:
            str: The converted list of strings, formatted using json.dumps.
    
        Example Output:
            "[\"3\", \"30\", \"34\"]"
        """
        str_list = list(map(str, numbers))
        return json.dumps(str_list)

    def CompareTwoElements(self, x: str, y: str):
        r"""
    
        Compare two strings to determine their order in forming the largest number.
        Return -1 if x should come before y; return 1 if y should come before x; return 0 if they are equal.
    
        Args:
            x (str): The first string to be compared.
            y (str): The second string to be compared.
    
        Returns:
            str: The comparison result, which is -1, 1, or 0.
    
        Example Output:
            "-1"
        """
        if x + y > y + x:
            return "-1"
        elif x + y < y + x:
            return "1"
        else:
            return "0"

    def SortList(self, arr: list, comparison_results: dict):
        r"""
    
        Sort the list of strings according to the comparison results to form the largest number.
    
        Args:
            arr (list[str]): The list of strings to be sorted.
            comparison_results (dict): A dictionary containing the comparison results between elements, with keys in the form of "x,y" strings and values as the comparison results.
    
        Returns:
            str: The sorted list of strings, formatted using json.dumps.
    
        Example Output:
            "[\"34\", \"3\", \"30\"]"
        """
        def compare_func(x, y):
            key = f"{x},{y}"
            return int(comparison_results.get(key, 0))
        
        sorted_arr = sorted(arr, key=cmp_to_key(compare_func))
        return json.dumps(sorted_arr)

    def ConcatenateList(self, arr: list):
        r"""
    
        Concatenate a list of strings into a single string.
    
        Args:
            arr (list[str]): The list of strings to be concatenated.
    
        Returns:
            str: The concatenated string.
    
        Example Output:
            "34330"
        """
        return ''.join(arr)

    def CheckLeadingZero(self, s: str):
        r"""
    
        Check if the string starts with a zero; if so, return "0", otherwise return the original string.
    
        Args:
            s (str): The string to be checked.
    
        Returns:
            str: The processed string.
    
        Example Output:
            "0"
        """
        if s and s[0] == '0':
            return "0"
        return s

    def Observe(self):
        r"""
    
        Return the list of integers in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of integers in the environment, formatted using json.dumps.
    
        Example Output:
            "[3, 30, 34]"
        """
        return json.dumps(self.arr)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 34330, Reference answer: 34330, Result: Correct, reward=1"
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
        numbers_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        numbers = json.loads(numbers_str)
        
        str_list_json = self.step(json.dumps({
            'name': 'ConvertToStrings',
            'parameters': {'numbers': numbers}
        }))[1]
        str_list = json.loads(str_list_json)
        
        comparison_results = {}
        n = len(str_list)
        for i in range(n):
            for j in range(n):
                if i != j:
                    x = str_list[i]
                    y = str_list[j]
                    key = f"{x},{y}"
                    res_str = self.step(json.dumps({
                        'name': 'CompareTwoElements',
                        'parameters': {'x': x, 'y': y}
                    }))[1]
                    comparison_results[key] = int(res_str)
        
        sorted_list_json = self.step(json.dumps({
            'name': 'SortList',
            'parameters': {
                'arr': str_list,
                'comparison_results': comparison_results
            }
        }))[1]
        sorted_list = json.loads(sorted_list_json)
        
        concatenated = self.step(json.dumps({
            'name': 'ConcatenateList',
            'parameters': {'arr': sorted_list}
        }))[1]
        
        final_answer = self.step(json.dumps({
            'name': 'CheckLeadingZero',
            'parameters': {'s': concatenated}
        }))[1]
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': final_answer}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_env_str1 = "LargestNumberEnv@{\"n\": 3, \"arr\": [3, 30, 34]}"
    env1 = LargestNumberEnv.from_env_str(test_env_str1)
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_env_str2 = "LargestNumberEnv@{\"n\": 5, \"arr\": [9, 89, 90, 91, 92]}"
    env2 = LargestNumberEnv.from_env_str(test_env_str2)
    print(env2.solve())
    print("step count:", env2.step_count)