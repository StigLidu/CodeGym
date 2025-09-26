# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestNonDecreasingSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_DP_ARRAY = 0
        self.COMPARE_CHARACTERS = 1
        self.UPDATE_DP_VALUE = 2
        self.FIND_MAX_DP_VALUE = 3
        self.OBSERVE = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeDPArray": self.INITIALIZE_DP_ARRAY,
            "CompareCharacters": self.COMPARE_CHARACTERS,
            "UpdateDPValue": self.UPDATE_DP_VALUE,
            "FindMaxDPValue": self.FIND_MAX_DP_VALUE,
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
        prefix = "LongestNonDecreasingSubsequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestNonDecreasingSubsequenceEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.dp = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.s:
            return 0
        
        n = len(self.s)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if self.s[i] >= self.s[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

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
            
            if action_code == self.INITIALIZE_DP_ARRAY:
                if "length" in params:
                    length = params["length"]
                    msg = self.InitializeDPArray(length)
                else:
                    msg = "Error: 'length' parameter is missing for INITIALIZE_DP_ARRAY action."
            
            elif action_code == self.COMPARE_CHARACTERS:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.CompareCharacters(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for COMPARE_CHARACTERS action."
                    
            elif action_code == self.UPDATE_DP_VALUE:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.UpdateDPValue(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for UPDATE_DP_VALUE action."
                    
            elif action_code == self.FIND_MAX_DP_VALUE:
                msg = self.FindMaxDPValue()
                
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
    def InitializeDPArray(self, length: int):
        r"""
    
        Initialize the DP array and set all elements to 1.
    
        Args:
            length (int): The length of the DP array, which is usually equal to the length of the string.
    
        Returns:
            str: The initialized DP array.
    
        Example Output:
            "[1, 1, 1, 1]"
        """
        self.dp = [1] * length
        return str(self.dp)

    def CompareCharacters(self, i: int, j: int):
        r"""
    
        Compare whether the characters at two positions in the string satisfy a non-decreasing relationship.
    
        Args:
            i (int): The index of the first character.
            j (int): The index of the second character.
    
        Returns:
            str: "True" means s[i] >= s[j], "False" means s[i] < s[j].
    
        Example Output:
            "True"
        """
        if i < 0 or i >= len(self.s) or j < 0 or j >= len(self.s):
            return "Error: Index out of range"
        return str(self.s[i] >= self.s[j])

    def UpdateDPValue(self, i: int, j: int):
        r"""
    
        Update the value at the specified position in the DP array based on the comparison result.
    
        Args:
            i (int): The index of the DP array to be updated.
            j (int): The index of the DP array used for comparison.
    
        Returns:
            str: The updated value of DP[i].
    
        Example Output:
            "3"
        """
        if i < 0 or i >= len(self.dp) or j < 0 or j >= len(self.dp):
            return "Error: Index out of range"
        if self.s[i] >= self.s[j]:
            self.dp[i] = max(self.dp[i], self.dp[j] + 1)
        return str(self.dp[i])

    def FindMaxDPValue(self):
        r"""
    
        Find the maximum value in the DP array, which is the length of the longest non-decreasing subsequence.
    
        Args:
            None
    
        Returns:
            str: The maximum value in the DP array.
    
        Example Output:
            "5"
        """
        if not self.dp:
            return "0"
        return str(max(self.dp))

    def Observe(self):
        r"""
    
        Obtain the string information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "abppplee"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the length of the longest non-decreasing subsequence.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 5, Reference answer: 5, Result: Correct, reward=1"
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
        length = len(s)
        self.step(json.dumps({'name': 'InitializeDPArray', 'parameters': {'length': length}}))
        for i in range(length):
            for j in range(i):
                compare_result = self.step(json.dumps({'name': 'CompareCharacters', 'parameters': {'i': i, 'j': j}}))[1]
                if compare_result == "True":
                    self.step(json.dumps({'name': 'UpdateDPValue', 'parameters': {'i': i, 'j': j}}))
        max_length = int(self.step(json.dumps({'name': 'FindMaxDPValue', 'parameters': {}}))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "abppplee"
    env = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"s\": \"{test_str}\"}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "zyxwvutsrqponmlkjihgfedcba"
    env = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"s\": \"{test_str}\"}}"
    )
    print(env.solve())
    print("step count:", env.step_count)