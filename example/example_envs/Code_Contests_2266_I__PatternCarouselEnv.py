# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class PatternCarouselEnv(gymnasium.Env):
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
        prefix = "PatternCarouselEnv@"
        if not env_str.startswith(prefix):
            return None
        return PatternCarouselEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.pattern = options.get("pattern", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        n = len(self.pattern)
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                if self.pattern[:i] * (n // i) == self.pattern:
                    return "YES"
        return "NO"

    # [Required] Define the step method of the environment
    def step(self, action: str):
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
                if "divisor" in params:
                    divisor = params["divisor"]
                    msg = self.CheckDivisibility(divisor)
                else:
                    msg = "Error: 'divisor' parameter is missing for CHECK_DIVISIBILITY action."
            
            elif action_code == self.GET_SUBSTRING:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.GetSubstring(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for GET_SUBSTRING action."
                    
            elif action_code == self.REPEAT_SUBSTRING:
                if "substring" in params and "times" in params:
                    substring = params["substring"]
                    times = params["times"]
                    msg = self.RepeatSubstring(substring, times)
                else:
                    msg = "Error: 'substring' or 'times' parameter is missing for REPEAT_SUBSTRING action."
                    
            elif action_code == self.COMPARE_STRINGS:
                if "str1" in params and "str2" in params:
                    str1 = params["str1"]
                    str2 = params["str2"]
                    msg = self.CompareStrings(str1, str2)
                else:
                    msg = "Error: 'str1' or 'str2' parameter is missing for COMPARE_STRINGS action."
                    
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
    
        Get the length of the current pattern string.
    
        Args:
            None
    
        Returns:
            str: The length of the pattern string.
    
        Example Output:
            "6"
        """
        return str(len(self.pattern))

    def CheckDivisibility(self, divisor: int):
        r"""
    
        Check if the length of the pattern string is divisible by the given divisor.
    
        Args:
            divisor (int): The divisor to check.
    
        Returns:
            str: Returns "True" if divisible, otherwise returns "False".
    
        Example Output:
            "True"
        """
        return str(len(self.pattern) % divisor == 0)

    def GetSubstring(self, start: int, end: int):
        r"""
    
        Get the substring from start to end (excluding end) in the pattern string.
    
        Args:
            start (int): The starting index of the substring.
            end (int): The ending index of the substring (excluding).
    
        Returns:
            str: The obtained substring.
    
        Example Output:
            "ab"
        """
        return self.pattern[start:end]

    def RepeatSubstring(self, substring: str, times: int):
        r"""
    
        Repeat the given substring for the specified number of times.
    
        Args:
            substring (str): The substring to be repeated.
            times (int): The number of times to repeat.
    
        Returns:
            str: The repeated string.
    
        Example Output:
            "ababab"
        """
        return substring * times

    def CompareStrings(self, str1: str, str2: str):
        r"""
    
        Compare whether two strings are equal.
    
        Args:
            str1 (str): The first string to compare.
            str2 (str): The second string to compare.
    
        Returns:
            str: Returns "True" if the two strings are equal, otherwise returns "False".
    
        Example Output:
            "True"
        """
        return str(str1 == str2)

    def Observe(self):
        r"""
    
        Get the current pattern string to be judged.
    
        Args:
            None
    
        Returns:
            str: The current pattern string.
    
        Example Output:
            "ababab"
        """
        return self.pattern

    def Done(self, answer: str):
        r"""
    
        Submit the final answer and verify if it is correct.
    
        Args:
            answer (str): The answer to submit, "YES" or "NO".
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: YES, Reference answer: YES, Result: Correct, reward=1"
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
        n = int(self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1])
        
        if n == 1:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'NO'}}))[1]
        
        for divisor in range(1, n//2 + 1):
            divisible = self.step(json.dumps({'name': 'CheckDivisibility', 'parameters': {'divisor': divisor}}))[1]
            if divisible == "True":
                times = n // divisor
                substring = self.step(json.dumps({'name': 'GetSubstring', 'parameters': {'start': 0, 'end': divisor}}))[1]
                repeated = self.step(json.dumps({'name': 'RepeatSubstring', 'parameters': {'substring': substring, 'times': times}}))[1]
                if self.step(json.dumps({'name': 'CompareStrings', 'parameters': {'str1': s, 'str2': repeated}}))[1] == "True":
                    return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'YES'}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'NO'}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_pattern = "ababab"
    env = PatternCarouselEnv.from_env_str(f"PatternCarouselEnv@{{\"pattern\": \"{test_pattern}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_pattern = "abac"
    env = PatternCarouselEnv.from_env_str(f"PatternCarouselEnv@{{\"pattern\": \"{test_pattern}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_pattern = "xxxx"
    env = PatternCarouselEnv.from_env_str(f"PatternCarouselEnv@{{\"pattern\": \"{test_pattern}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 4
    print("\nTest Case 4:")
    test_pattern = "hellohellohello"
    env = PatternCarouselEnv.from_env_str(f"PatternCarouselEnv@{{\"pattern\": \"{test_pattern}\"}}")
    print(env.solve())
    print("step count:", env.step_count)