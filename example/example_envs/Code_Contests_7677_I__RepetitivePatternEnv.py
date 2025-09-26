# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class RepetitivePatternEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GET_STRING_LENGTH = 0
        self.CHECK_DIVISIBILITY = 1
        self.GET_SUBSTRING = 2
        self.REPEAT_SUBSTRING = 3
        self.COMPARE_STRINGS = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "GetStringLength": self.GET_STRING_LENGTH,
            "CheckDivisibility": self.CHECK_DIVISIBILITY,
            "GetSubstring": self.GET_SUBSTRING,
            "RepeatSubstring": self.REPEAT_SUBSTRING,
            "CompareStrings": self.COMPARE_STRINGS,
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
        prefix = "RepetitivePatternEnv@"
        if not env_str.startswith(prefix):
            return None
        return RepetitivePatternEnv(env_str=env_str)

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
        for i in range(1, n // 2 + 1):
            if n % i == 0 and self.s[:i] * (n // i) == self.s:
                return True
        return False

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
            
            if action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
            
            elif action_code == self.CHECK_DIVISIBILITY:
                if "n" in params and "i" in params:
                    n = params["n"]
                    i = params["i"]
                    msg = self.CheckDivisibility(n, i)
                else:
                    msg = "Error: 'n' or 'i' parameter is missing for CHECK_DIVISIBILITY action."
            
            elif action_code == self.GET_SUBSTRING:
                if "i" in params:
                    i = params["i"]
                    msg = self.GetSubstring(i)
                else:
                    msg = "Error: 'i' parameter is missing for GET_SUBSTRING action."
            
            elif action_code == self.REPEAT_SUBSTRING:
                if "substr" in params and "times" in params:
                    substr = params["substr"]
                    times = params["times"]
                    msg = self.RepeatSubstring(substr, times)
                else:
                    msg = "Error: 'substr' or 'times' parameter is missing for REPEAT_SUBSTRING action."
            
            elif action_code == self.COMPARE_STRINGS:
                if "s1" in params and "s2" in params:
                    s1 = params["s1"]
                    s2 = params["s2"]
                    msg = self.CompareStrings(s1, s2)
                else:
                    msg = "Error: 's1' or 's2' parameter is missing for COMPARE_STRINGS action."
            
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
    def GetStringLength(self):
        r"""
    
        Get the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the string.
    
        Example Output:
            "4"
        """
        return str(len(self.s))

    def CheckDivisibility(self, n: int, i: int):
        r"""
    
        Check if n is divisible by i.
    
        Args:
            n (int): The dividend.
            i (int): The divisor.
    
        Returns:
            str: "true" if n is divisible by i, otherwise "false".
    
        Example Output:
            "true"
        """
        return "true" if n % i == 0 else "false"

    def GetSubstring(self, i: int):
        r"""
    
        Get the first i characters of the string as a substring.
    
        Args:
            i (int): The length of the substring.
    
        Returns:
            str: The substring consisting of the first i characters.
    
        Example Output:
            "ab"
        """
        return self.s[:i]

    def RepeatSubstring(self, substr: str, times: int):
        r"""
    
        Repeat the substring for the specified number of times.
    
        Args:
            substr (str): The substring to be repeated.
            times (int): The number of repetitions.
    
        Returns:
            str: The string after repetition.
    
        Example Output:
            "abab"
        """
        return substr * times

    def CompareStrings(self, s1: str, s2: str):
        r"""
    
        Compare whether two strings are equal.
    
        Args:
            s1 (str): The first string.
            s2 (str): The second string.
    
        Returns:
            str: "true" if the two strings are equal, otherwise "false".
    
        Example Output:
            "true"
        """
        return "true" if s1 == s2 else "false"

    def Observe(self):
        r"""
    
        Get the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "abab"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (bool): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: true, Reference answer: true, Result: Correct, reward=1"
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
        n_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        n = int(n_str)
        
        if n < 2:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
        
        for i in range(1, n//2 + 1):
            divisible = self.step(json.dumps({'name': 'CheckDivisibility', 'parameters': {'n': n, 'i': i}}))[1]
            if divisible == "true":
                substr = self.step(json.dumps({'name': 'GetSubstring', 'parameters': {'i': i}}))[1]
                times = n // i
                repeated = self.step(json.dumps({'name': 'RepeatSubstring', 'parameters': {'substr': substr, 'times': times}}))[1]
                equal = self.step(json.dumps({'name': 'CompareStrings', 'parameters': {'s1': repeated, 's2': s}}))[1]
                if equal == "true":
                    return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': True}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "abab"
    env = RepetitivePatternEnv.from_env_str(f"RepetitivePatternEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "abcd"
    env = RepetitivePatternEnv.from_env_str(f"RepetitivePatternEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 3
    print("\nTest Case 3:")
    test_string = "abcabcabc"
    env = RepetitivePatternEnv.from_env_str(f"RepetitivePatternEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)