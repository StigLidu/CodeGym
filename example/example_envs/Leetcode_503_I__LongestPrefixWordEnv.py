# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestPrefixWordEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_PREFIX_EXISTS = 1
        self.GET_WORD_PREFIX = 2
        self.COMPARE_WORD_LENGTHS = 3
        self.GET_WORD_INDEX = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckPrefixExists": self.CHECK_PREFIX_EXISTS,
            "GetWordPrefix": self.GET_WORD_PREFIX,
            "CompareWordLengths": self.COMPARE_WORD_LENGTHS,
            "GetWordIndex": self.GET_WORD_INDEX,
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
        prefix = "LongestPrefixWordEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestPrefixWordEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.words = options.get("words", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        words_set = set(self.words)
        longest_word = ""

        for word in self.words:
            for i in range(1, len(word)):
                if word[:i] in words_set:
                    if len(word[:i]) > len(longest_word):
                        longest_word = word[:i]
                    elif len(word[:i]) == len(longest_word):
                        if self.words.index(word[:i]) < self.words.index(longest_word):
                            longest_word = word[:i]
                    
        return longest_word

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
                
            elif action_code == self.CHECK_PREFIX_EXISTS:
                if "prefix" in params:
                    prefix = params["prefix"]
                    msg = self.CheckPrefixExists(prefix)
                else:
                    msg = "Error: 'prefix' parameter is missing for CHECK_PREFIX_EXISTS action."
                    
            elif action_code == self.GET_WORD_PREFIX:
                if "word" in params and "length" in params:
                    word = params["word"]
                    length = params["length"]
                    msg = self.GetWordPrefix(word, length)
                else:
                    msg = "Error: 'word' or 'length' parameter is missing for GET_WORD_PREFIX action."
                    
            elif action_code == self.COMPARE_WORD_LENGTHS:
                if "word1" in params and "word2" in params:
                    word1 = params["word1"]
                    word2 = params["word2"]
                    msg = self.CompareWordLengths(word1, word2)
                else:
                    msg = "Error: 'word1' or 'word2' parameter is missing for COMPARE_WORD_LENGTHS action."
                    
            elif action_code == self.GET_WORD_INDEX:
                if "word" in params:
                    word = params["word"]
                    msg = self.GetWordIndex(word)
                else:
                    msg = "Error: 'word' parameter is missing for GET_WORD_INDEX action."
                    
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
    
        Obtain the list of words in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of words in the current environment, represented in JSON format.
    
        Example Output:
            "[\"a\", \"app\", \"apple\", \"banana\"]"
        """
        return json.dumps(self.words)

    def CheckPrefixExists(self, prefix: str):
        r"""
    
        Check if the specified prefix exists in the word list.
    
        Args:
            prefix (str): The prefix to be checked.
    
        Returns:
            str: "True" indicates existence, "False" indicates non-existence.
    
        Example Output:
            "True"
        """
        return str(prefix in self.words)

    def GetWordPrefix(self, word: str, length: int):
        r"""
    
        Obtain the prefix of the specified length for the given word.
    
        Args:
            word (str): The original word.
            length (int): The length of the prefix.
    
        Returns:
            str: The prefix of the word. If the length exceeds the length of the word, return an empty string.
    
        Example Output:
            "app"
        """
        if length <= 0 or length > len(word):
            return ""
        return word[:length]

    def CompareWordLengths(self, word1: str, word2: str):
        r"""
    
        Compare the lengths of two words.
    
        Args:
            word1 (str): The first word.
            word2 (str): The second word.
    
        Returns:
            str: "1" means word1 is longer, "-1" means word2 is longer, "0" means the lengths are equal.
    
        Example Output:
            "1"
        """
        if len(word1) > len(word2):
            return "1"
        elif len(word1) < len(word2):
            return "-1"
        else:
            return "0"

    def GetWordIndex(self, word: str):
        r"""
    
        Obtain the index of the word in the original list.
    
        Args:
            word (str): The word whose index needs to be found.
    
        Returns:
            str: The index of the word in the list. If it does not exist, return "-1".
    
        Example Output:
            "1"
        """
        try:
            return str(self.words.index(word))
        except ValueError:
            return "-1"

    def Done(self, answer: str):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: app, Reference answer: app, Result: Correct, reward=1"
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
        import json
        words_json = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        words = json.loads(words_json)
        
        candidate = ""
        for word in words:
            if candidate:
                comp_res = self.step(json.dumps({
                    'name': 'CompareWordLengths',
                    'parameters': {'word1': word, 'word2': candidate}
                }))[1]
                if comp_res == "-1":
                    continue
                elif comp_res == "0":
                    current_idx = int(self.step(json.dumps({
                        'name': 'GetWordIndex',
                        'parameters': {'word': word}
                    }))[1])
                    candidate_idx = int(self.step(json.dumps({
                        'name': 'GetWordIndex',
                        'parameters': {'word': candidate}
                    }))[1])
                    if current_idx > candidate_idx:
                        continue
            
            is_valid = False
            for other_word in words:
                if word == other_word:
                    continue
                prefix = self.step(json.dumps({
                    'name': 'GetWordPrefix',
                    'parameters': {'word': other_word, 'length': len(word)}
                }))[1]
                if prefix == word:
                    is_valid = True
                    break
            
            if is_valid:
                candidate = word
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': candidate}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_words1 = ["a", "app", "apple", "application", "banana", "applet"]
    env1 = LongestPrefixWordEnv.from_env_str(f"LongestPrefixWordEnv@{{\"words\": {test_words1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_words2 = ["cat", "dog", "bird", "cathedral", "dolphin", ""]
    env2 = LongestPrefixWordEnv.from_env_str(f"LongestPrefixWordEnv@{{\"words\": {test_words2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)