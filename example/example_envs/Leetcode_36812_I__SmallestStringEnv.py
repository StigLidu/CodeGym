# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class SmallestStringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GENERATE_CANDIDATE = 0
        self.COMPARE_STRINGS = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "GenerateCandidate": self.GENERATE_CANDIDATE,
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
        prefix = "SmallestStringEnv@"
        if not env_str.startswith(prefix):
            return None
        return SmallestStringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.word = options.get("word", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if len(self.word) <= 1:
            return ""
        smallest_string = self.word[1:]  # Initialize with first character removed
        for i in range(len(self.word)):
            candidate = self.word[:i] + self.word[i+1:]
            if candidate < smallest_string:
                smallest_string = candidate
        return smallest_string

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
            
            if action_code == self.GENERATE_CANDIDATE:
                if "index" in params:
                    index = params["index"]
                    msg = self.GenerateCandidate(index)
                else:
                    msg = "Error: 'index' parameter is missing for GENERATE_CANDIDATE action."
            
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
    def GenerateCandidate(self, index: int):
        r"""
    
        Generates a candidate string after deleting the character at the specified index position.
    
        Args:
            index (int): The index of the character to be deleted.
    
        Returns:
            str: The string after deleting the character at the specified index.
    
        Example Output:
            "acd"
        """
        if 0 <= index < len(self.word):
            return self.word[:index] + self.word[index+1:]
        return ""

    def CompareStrings(self, s1: str, s2: str):
        r"""
    
        Compares the lexicographical order of two strings.
    
        Args:
            s1 (str): The first string to be compared.
            s2 (str): The second string to be compared.
    
        Returns:
            str: Returns "-1" if s1 < s2, "1" if s1 > s2, and "0" if they are equal.
    
        Example Output:
            "-1"
        """
        if s1 < s2:
            return "-1"
        elif s1 > s2:
            return "1"
        else:
            return "0"

    def Observe(self):
        r"""
    
        Obtains the original string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The original string in the current environment.
    
        Example Output:
            "abcd"
        """
        return self.word

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: acd, Reference answer: acd, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
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
        original_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = len(original_str)
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': ''}}))[1]
        
        min_candidate = self.step(json.dumps({'name': 'GenerateCandidate', 'parameters': {'index': 0}}))[1]
        
        for i in range(1, n):
            current_candidate = self.step(json.dumps({'name': 'GenerateCandidate', 'parameters': {'index': i}}))[1]
            compare_result = self.step(json.dumps({'name': 'CompareStrings', 'parameters': {'s1': current_candidate, 's2': min_candidate}}))[1]
            if compare_result == "-1":
                min_candidate = current_candidate
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': min_candidate}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_word = "cbacdcbc"
    env = SmallestStringEnv.from_env_str(f"SmallestStringEnv@{{\"word\": \"{test_word}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_word = "abcd"
    env = SmallestStringEnv.from_env_str(f"SmallestStringEnv@{{\"word\": \"{test_word}\"}}")
    print(env.solve())
    print("step count:", env.step_count)