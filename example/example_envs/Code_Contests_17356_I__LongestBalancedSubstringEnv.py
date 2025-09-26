# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class LongestBalancedSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.IS_VOWEL = 1
        self.COUNT_VOWELS_CONSONANTS = 2
        self.UPDATE_MAX_LENGTH = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "IsVowel": self.IS_VOWEL,
            "CountVowelsConsonants": self.COUNT_VOWELS_CONSONANTS,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property of the environment
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "LongestBalancedSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestBalancedSubstringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.max_length = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        vowels_set = set('aeiou')
        max_length = 0
        
        for i in range(len(self.s)):
            vowels = 0
            consonants = 0
            for j in range(i, len(self.s)):
                if self.s[j] in vowels_set:
                    vowels += 1
                else:
                    consonants += 1
                if vowels == consonants:
                    max_length = max(max_length, j - i + 1)
        
        return max_length if max_length > 0 else -1

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
            
            if action_code == self.OBSERVE:
                msg = self.Observe()
            
            elif action_code == self.IS_VOWEL:
                if "char" in params:
                    char = params["char"]
                    msg = self.IsVowel(char)
                else:
                    msg = "Error: 'char' parameter is missing for IS_VOWEL action."
            
            elif action_code == self.COUNT_VOWELS_CONSONANTS:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CountVowelsConsonants(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for COUNT_VOWELS_CONSONANTS action."
            
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
    
        Returns the string information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current string to be analyzed.
    
        Example Output:
            "abcde"
        """
        return self.s

    def IsVowel(self, char: str):
        r"""
    
        Determines whether the given character is a vowel (a, e, i, o, u).
    
        Args:
            char (str): The single character to be judged.
    
        Returns:
            str: "True" indicates it is a vowel, "False" indicates it is not a vowel.
    
        Example Output:
            "True"
        """
        return str(char in 'aeiou')

    def CountVowelsConsonants(self, start: int, end: int):
        r"""
    
        Counts the number of vowels and consonants in the substring from start to end (inclusive of end).
    
        Args:
            start (int): The starting index of the substring (0-based).
            end (int): The ending index of the substring (0-based).
    
        Returns:
            str: A JSON string containing the number of vowels and consonants.
    
        Example Output:
            "{\"vowels\": 2, \"consonants\": 3}"
        """
        if start < 0 or end >= len(self.s) or start > end:
            return json.dumps({"vowels": 0, "consonants": 0})
        
        vowels = 0
        consonants = 0
        for i in range(start, end + 1):
            if self.s[i] in 'aeiou':
                vowels += 1
            else:
                consonants += 1
        
        return json.dumps({"vowels": vowels, "consonants": consonants})

    def UpdateMaxLength(self, current_length: int):
        r"""
    
        Updates the length of the longest balanced substring found so far.
    
        Args:
            current_length (int): The length of the balanced substring currently being checked.
    
        Returns:
            str: The updated maximum length.
    
        Example Output:
            "6"
        """
        if current_length > self.max_length:
            self.max_length = current_length
        return str(self.max_length)

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (int): The length of the longest balanced substring submitted by the user, -1 if it does not exist.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
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
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = len(s)
        if n < 2:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': -1}}))[1]
        
        max_len = -1
        for start in range(n):
            for end in range(start, n):
                count_json = self.step(json.dumps({
                    'name': 'CountVowelsConsonants',
                    'parameters': {'start': start, 'end': end}
                }))[1]
                count_dict = json.loads(count_json)
                vowels = count_dict['vowels']
                consonants = count_dict['consonants']
                if vowels == consonants and vowels > 0:
                    current_length = end - start + 1
                    updated_max = int(self.step(json.dumps({
                        'name': 'UpdateMaxLength',
                        'parameters': {'current_length': current_length}
                    }))[1])
                    if updated_max > max_len:
                        max_len = updated_max
        
        final_answer = max_len if max_len != -1 else -1
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "abcde"
    env = LongestBalancedSubstringEnv.from_env_str(f"LongestBalancedSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "aeiouxyz"
    env = LongestBalancedSubstringEnv.from_env_str(f"LongestBalancedSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 3
    print("\nTest Case 3:")
    test_str = "abecidofu"
    env = LongestBalancedSubstringEnv.from_env_str(f"LongestBalancedSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)