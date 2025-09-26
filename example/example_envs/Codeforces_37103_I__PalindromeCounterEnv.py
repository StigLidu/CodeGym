# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import re
import ast
import json

class PalindromeCounterEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CLEAN_PHRASE = 0
        self.CHECK_PALINDROME = 1
        self.COUNT_PALINDROMES = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CleanPhrase": self.CLEAN_PHRASE,
            "CheckPalindrome": self.CHECK_PALINDROME,
            "CountPalindromes": self.COUNT_PALINDROMES,
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
        prefix = "PalindromeCounterEnv@"
        if not env_str.startswith(prefix):
            return None
        return PalindromeCounterEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.input_data = options.get("input_data", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def is_palindrome(phrase):
            cleaned_phrase = re.sub(r'[^A-Za-z0-9]', '', phrase).lower()
            return cleaned_phrase == cleaned_phrase[::-1]
        
        t = int(self.input_data[0])
        phrases = self.input_data[1:t+1]
        return sum(is_palindrome(phrase) for phrase in phrases)

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
            
            if action_code == self.CLEAN_PHRASE:
                if "phrase" in params:
                    phrase = params["phrase"]
                    msg = self.CleanPhrase(phrase)
                else:
                    msg = "Error: 'phrase' parameter is missing for CLEAN_PHRASE action."
            
            elif action_code == self.CHECK_PALINDROME:
                if "cleaned_phrase" in params:
                    cleaned_phrase = params["cleaned_phrase"]
                    msg = self.CheckPalindrome(cleaned_phrase)
                else:
                    msg = "Error: 'cleaned_phrase' parameter is missing for CHECK_PALINDROME action."
                    
            elif action_code == self.COUNT_PALINDROMES:
                if "results" in params:
                    results = params["results"]
                    msg = self.CountPalindromes(results)
                else:
                    msg = "Error: 'results' parameter is missing for COUNT_PALINDROMES action."
                    
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
    def CleanPhrase(self, phrase: str):
        r"""
    
        Clean the phrase by removing non-alphanumeric characters and converting to lowercase.
    
        Args:
            phrase (str): The phrase to be cleaned.
    
        Returns:
            str: The cleaned phrase.
    
        Example Output:
            "amanaplanacanalpanama"
        """
        cleaned_phrase = re.sub(r'[^A-Za-z0-9]', '', phrase).lower()
        return cleaned_phrase

    def CheckPalindrome(self, cleaned_phrase: str):
        r"""
    
        Check if the cleaned phrase is a palindrome.
    
        Args:
            cleaned_phrase (str): The cleaned phrase.
    
        Returns:
            str: "True" indicates it is a palindrome, "False" indicates it is not.
    
        Example Output:
            "True"
        """
        is_pal = cleaned_phrase == cleaned_phrase[::-1]
        return str(is_pal)

    def CountPalindromes(self, results: list):
        r"""
    
        Count the number of palindromes.
    
        Args:
            results (list[str]): A list of judgment results indicating whether each phrase is a palindrome, with elements "True" or "False".
    
        Returns:
            str: The number of palindromes.
    
        Example Output:
            "2"
        """
        count = sum(1 for res in results if res == "True")
        return str(count)

    def Observe(self):
        r"""
    
        Obtain the input data in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string representation of the input data.
    
        Example Output:
            "['4', 'A man, a plan, a canal, Panama', 'Not a palindrome', 'Madam In Eden, I'm Adam', 'Hello, World!']"
        """
        return str(self.input_data)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The user-submitted answer for the number of palindromes.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        input_data = ast.literal_eval(observe_result)
        num_phrases = int(input_data[0])
        phrases = input_data[1:1+num_phrases]
        
        results = []
        for phrase in phrases:
            cleaned = self.step(json.dumps({'name': 'CleanPhrase', 'parameters': {'phrase': phrase}}))[1]
            is_palindrome = self.step(json.dumps({'name': 'CheckPalindrome', 'parameters': {'cleaned_phrase': cleaned}}))[1]
            results.append(is_palindrome)
        
        palindrome_count = int(self.step(json.dumps({'name': 'CountPalindromes', 'parameters': {'results': results}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': palindrome_count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_input1 = [
        "4",
        "A man, a plan, a canal, Panama",
        "Not a palindrome",
        "Madam In Eden, I'm Adam",
        "Hello, World!"
    ]
    env1 = PalindromeCounterEnv.from_env_str(f"PalindromeCounterEnv@{{\"input_data\": {test_input1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_input2 = [
        "3",
        "Racecar",
        "Python Programming",
        "Able was I ere I saw Elba"
    ]
    env2 = PalindromeCounterEnv.from_env_str(f"PalindromeCounterEnv@{{\"input_data\": {test_input2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)