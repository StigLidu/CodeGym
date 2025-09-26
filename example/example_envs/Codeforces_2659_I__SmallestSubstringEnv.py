# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from collections import defaultdict

class SmallestSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GET_UNIQUE_CHARS = 0
        self.GET_STRING_LENGTH = 1
        self.GET_CHAR_AT_POSITION = 2
        self.INCREMENT_COUNTER = 3
        self.DECREMENT_COUNTER = 4
        self.GET_COUNTER_VALUE = 5
        self.UPDATE_MIN_LENGTH = 6
        self.OBSERVE = 7
        self.DONE = 8

        # [Required] Define the action mapping
        self.func_mapping = {
            "GetUniqueChars": self.GET_UNIQUE_CHARS,
            "GetStringLength": self.GET_STRING_LENGTH,
            "GetCharAtPosition": self.GET_CHAR_AT_POSITION,
            "IncrementCounter": self.INCREMENT_COUNTER,
            "DecrementCounter": self.DECREMENT_COUNTER,
            "GetCounterValue": self.GET_COUNTER_VALUE,
            "UpdateMinLength": self.UPDATE_MIN_LENGTH,
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
        prefix = "SmallestSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return SmallestSubstringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        
        # Initialize counters for tracking character occurrences
        self.counters = defaultdict(int)
        self.formed = 0
        self.required_len = len(set(self.s)) if self.s else 0
        self.min_length = float('inf')
        
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.s)
        if n == 0:
            return 0
            
        unique_chars = set(self.s)
        required_len = len(unique_chars)
        min_length = float('inf')
        
        left = 0
        unique_char_count = defaultdict(int)
        formed = 0

        for right in range(n):
            unique_char_count[self.s[right]] += 1
            if unique_char_count[self.s[right]] == 1:
                formed += 1
                
            while formed == required_len:
                min_length = min(min_length, right - left + 1)
                unique_char_count[self.s[left]] -= 1
                if unique_char_count[self.s[left]] == 0:
                    formed -= 1
                left += 1

        return min_length if min_length != float('inf') else 0

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
            msg = ""
            
            if action_code == self.GET_UNIQUE_CHARS:
                msg = self.GetUniqueChars()
            
            elif action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
                
            elif action_code == self.GET_CHAR_AT_POSITION:
                if "position" in params:
                    position = params["position"]
                    msg = self.GetCharAtPosition(position)
                else:
                    msg = "Error: 'position' parameter is missing for GET_CHAR_AT_POSITION action."
                    
            elif action_code == self.INCREMENT_COUNTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.IncrementCounter(char)
                else:
                    msg = "Error: 'char' parameter is missing for INCREMENT_COUNTER action."
                    
            elif action_code == self.DECREMENT_COUNTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.DecrementCounter(char)
                else:
                    msg = "Error: 'char' parameter is missing for DECREMENT_COUNTER action."
                    
            elif action_code == self.GET_COUNTER_VALUE:
                if "char" in params:
                    char = params["char"]
                    msg = self.GetCounterValue(char)
                else:
                    msg = "Error: 'char' parameter is missing for GET_COUNTER_VALUE action."
                    
            elif action_code == self.UPDATE_MIN_LENGTH:
                if "current_length" in params:
                    current_length = params["current_length"]
                    msg = self.UpdateMinLength(current_length)
                else:
                    msg = "Error: 'current_length' parameter is missing for UPDATE_MIN_LENGTH action."
                    
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
    
        Obtain the set of all unique characters in the string.
    
        Args:
            None
    
        Returns:
            str: A string representation of the set containing all unique characters.
    
        Example Output:
            "{'a', 'b', 'c'}"
        """
        unique_chars = set(self.s)
        return str(unique_chars)

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

    def GetCharAtPosition(self, position: int):
        r"""
    
        Get the character at the specified position in the string.
    
        Args:
            position (int): The position index of the character to retrieve.
    
        Returns:
            str: The character at the specified position, or an error message if the position is invalid.
    
        Example Output:
            "a"
        """
        if 0 <= position < len(self.s):
            return self.s[position]
        return "Error: Invalid position"

    def IncrementCounter(self, char: str):
        r"""
    
        Increase the counter value of the specified character. If the counter changes from 0 to 1, increment the formed count.
    
        Args:
            char (str): The character whose count is to be increased.
    
        Returns:
            str: The updated counter value and formed value.
    
        Example Output:
            "Counter for 'a': 1, formed: 1"
        """
        prev_count = self.counters[char]
        self.counters[char] += 1
        
        if prev_count == 0:
            self.formed += 1
            
        return f"Counter for '{char}': {self.counters[char]}, formed: {self.formed}"

    def DecrementCounter(self, char: str):
        r"""
    
        Decrease the counter value of the specified character. If the counter changes from 1 to 0, decrement the formed count.
    
        Args:
            char (str): The character whose count is to be decreased.
    
        Returns:
            str: The updated counter value and formed value.
    
        Example Output:
            "Counter for 'a': 0, formed: 0"
        """
        if self.counters[char] > 0:
            prev_count = self.counters[char]
            self.counters[char] -= 1
            
            if prev_count == 1:
                self.formed -= 1
                
            return f"Counter for '{char}': {self.counters[char]}, formed: {self.formed}"
        return f"Counter for '{char}': 0, formed: {self.formed}"

    def GetCounterValue(self, char: str):
        r"""
    
        Get the current counter value of the specified character.
    
        Args:
            char (str): The character to query.
    
        Returns:
            str: The counter value of the specified character.
    
        Example Output:
            "2"
        """
        return str(self.counters[char])

    def UpdateMinLength(self, current_length: int):
        r"""
    
        Update the minimum length if the current length is less than the current minimum length.
    
        Args:
            current_length (int): The length of the current window.
    
        Returns:
            str: The updated minimum length.
    
        Example Output:
            "3"
        """
        if current_length < self.min_length:
            self.min_length = current_length
        return str(self.min_length)

    def Observe(self):
        r"""
    
        Return the observation information of the current environment, including the formed value, required_len value, and current min_length value.
    
        Args:
            None
    
        Returns:
            str: Information describing the current state of the environment.
    
        Example Output:
            "formed: 2, required_len: 3, min_length: inf"
        """
        return f"formed: {self.formed}, required_len: {self.required_len}, min_length: {self.min_length}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the length of the minimum substring.
    
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
        import ast
    
        unique_chars_str = self.step(json.dumps({'name': 'GetUniqueChars', 'parameters': {}}))[1]
        unique_chars = ast.literal_eval(unique_chars_str)
        required_len = len(unique_chars)
        if required_len == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
    
        str_len_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        str_len = int(str_len_str)
        if str_len == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
    
        left = 0
        right = 0
    
        while right < str_len:
            current_char = self.step(json.dumps({
                'name': 'GetCharAtPosition',
                'parameters': {'position': right}
            }))[1]
            self.step(json.dumps({
                'name': 'IncrementCounter',
                'parameters': {'char': current_char}
            }))
            right += 1
    
            while True:
                observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
                parts = observe_info.split(', ')
                formed = int(parts[0].split(': ')[1])
                current_min = parts[2].split(': ')[1]
                current_min = float('inf') if current_min == 'inf' else int(current_min)
    
                if formed == required_len:
                    current_window_len = right - left
                    new_min = self.step(json.dumps({
                        'name': 'UpdateMinLength',
                        'parameters': {'current_length': current_window_len}
                    }))[1]
                    left_char = self.step(json.dumps({
                        'name': 'GetCharAtPosition',
                        'parameters': {'position': left}
                    }))[1]
                    self.step(json.dumps({
                        'name': 'DecrementCounter',
                        'parameters': {'char': left_char}
                    }))
                    left += 1
                else:
                    break
    
        final_observe = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        final_min = int(final_observe.split(', ')[2].split(': ')[1])
    
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_min}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "abac"
    env = SmallestSubstringEnv.from_env_str(f"SmallestSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "aabcbcdbca"
    env = SmallestSubstringEnv.from_env_str(f"SmallestSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_str = "abcd"
    env = SmallestSubstringEnv.from_env_str(f"SmallestSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)