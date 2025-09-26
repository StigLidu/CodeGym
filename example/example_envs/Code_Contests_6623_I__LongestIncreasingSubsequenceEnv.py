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

class LongestIncreasingSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_DP_ARRAY = 0
        self.COMPARE_ELEMENTS = 1
        self.UPDATE_DP_VALUE = 2
        self.FIND_MAX_VALUE = 3
        self.OBSERVE = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeDpArray": self.INITIALIZE_DP_ARRAY,
            "CompareElements": self.COMPARE_ELEMENTS,
            "UpdateDpValue": self.UPDATE_DP_VALUE,
            "FindMaxValue": self.FIND_MAX_VALUE,
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
        prefix = "LongestIncreasingSubsequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestIncreasingSubsequenceEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.sequence = options.get("sequence", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.sequence:
            return 0

        dp = [1] * len(self.sequence)

        for i in range(1, len(self.sequence)):
            for j in range(0, i):
                if self.sequence[i] > self.sequence[j]:
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
                    msg = self.InitializeDpArray(length)
                else:
                    msg = "Error: 'length' parameter is missing for INITIALIZE_DP_ARRAY action."
            
            elif action_code == self.COMPARE_ELEMENTS:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.CompareElements(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for COMPARE_ELEMENTS action."
                    
            elif action_code == self.UPDATE_DP_VALUE:
                if "i" in params and "j" in params and "dp" in params:
                    i = params["i"]
                    j = params["j"]
                    dp = params["dp"]
                    msg = self.UpdateDpValue(i, j, dp)
                else:
                    msg = "Error: 'i', 'j' or 'dp' parameter is missing for UPDATE_DP_VALUE action."
                    
            elif action_code == self.FIND_MAX_VALUE:
                if "array" in params:
                    array = params["array"]
                    msg = self.FindMaxValue(array)
                else:
                    msg = "Error: 'array' parameter is missing for FIND_MAX_VALUE action."
                    
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
    def InitializeDpArray(self, length: int):
        r"""
    
        Initialize a dp array of the specified length, with all elements set to 1.
    
        Args:
            length (int): The length of the dp array.
    
        Returns:
            str: The initialized dp array, represented as a JSON format string.
    
        Example Output:
            "[1, 1, 1, 1, 1]"
        """
        dp_array = [1] * length
        return json.dumps(dp_array)

    def CompareElements(self, i: int, j: int):
        r"""
    
        Compare the sizes of the i-th and j-th elements in the sequence.
    
        Args:
            i (int): The index of the i-th element in the sequence.
            j (int): The index of the j-th element in the sequence.
    
        Returns:
            str: Returns "True" if sequence[i] > sequence[j], otherwise returns "False".
    
        Example Output:
            "True"
        """
        if i < 0 or i >= len(self.sequence) or j < 0 or j >= len(self.sequence):
            return "False"
        return str(self.sequence[i] > self.sequence[j])

    def UpdateDpValue(self, i: int, j: int, dp: list):
        r"""
    
        Update the value of dp[i] based on the value of dp[j]. If dp[j] + 1 is greater than the current value of dp[i], then update it.
    
        Args:
            i (int): The index of the position to be updated in the dp array.
            j (int): The index of the position in the dp array used for comparison.
            dp (list): The current dp array.
    
        Returns:
            str: The updated dp array, represented as a JSON format string.
    
        Example Output:
            "[1, 1, 2, 1, 1]"
        """
        if i < 0 or i >= len(dp) or j < 0 or j >= len(dp):
            return json.dumps(dp)
            
        if dp[j] + 1 > dp[i]:
            dp[i] = dp[j] + 1
        return json.dumps(dp)

    def FindMaxValue(self, array: list):
        r"""
    
        Find the maximum value in the array.
    
        Args:
            array (list): The array to find the maximum value from.
    
        Returns:
            str: The maximum value in the array.
    
        Example Output:
            "5"
        """
        if not array:
            return "0"
        return str(max(array))

    def Observe(self):
        r"""
    
        Obtain the sequence information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The sequence in the current environment, represented as a JSON format string.
    
        Example Output:
            "[5, 2, 7, 4, 3, 8]"
        """
        return json.dumps(self.sequence)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the length of the longest strictly increasing subsequence.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
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
        sequence_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        sequence = json.loads(sequence_str)
        n = len(sequence)
        
        dp_str = self.step(json.dumps({'name': 'InitializeDpArray', 'parameters': {'length': n}}))[1]
        dp = json.loads(dp_str)
        
        for i in range(n):
            for j in range(i):
                compare_result = self.step(json.dumps({'name': 'CompareElements', 'parameters': {'i': i, 'j': j}}))[1]
                if compare_result == "True":
                    dp_str = self.step(json.dumps({'name': 'UpdateDpValue', 'parameters': {'i': i, 'j': j, 'dp': dp}}))[1]
                    dp = json.loads(dp_str)
        
        max_length = int(self.step(json.dumps({'name': 'FindMaxValue', 'parameters': {'array': dp}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_sequence = [5, 2, 7, 4, 3, 8]
    env = LongestIncreasingSubsequenceEnv.from_env_str(
        f"LongestIncreasingSubsequenceEnv@{{\"sequence\": {test_sequence}}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_sequence = [10, 22, 9, 33, 21, 50, 41, 60]
    env = LongestIncreasingSubsequenceEnv.from_env_str(
        f"LongestIncreasingSubsequenceEnv@{{\"sequence\": {test_sequence}}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 3 (random)
    print("\nTest Case 3 (Random):")
    test_sequence = [random.randint(1, 100) for _ in range(random.randint(5, 15))]
    env = LongestIncreasingSubsequenceEnv.from_env_str(
        f"LongestIncreasingSubsequenceEnv@{{\"sequence\": {test_sequence}}}"
    )
    print(f"Sequence: {test_sequence}")
    print(env.solve())
    print("step count:", env.step_count)