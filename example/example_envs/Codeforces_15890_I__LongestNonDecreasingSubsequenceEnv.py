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

class LongestNonDecreasingSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_DP_ARRAY = 0
        self.UPDATE_DP_VALUE = 1
        self.FIND_MAX_DP_VALUE = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeDPArray": self.INITIALIZE_DP_ARRAY,
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
        self.heights = options.get("heights", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.heights:
            return 0

        n = len(self.heights)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if self.heights[i] >= self.heights[j]:
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
            
            elif action_code == self.UPDATE_DP_VALUE:
                if "i" in params and "j" in params and "dp_array" in params and "heights" in params:
                    i = params["i"]
                    j = params["j"]
                    dp_array = params["dp_array"]
                    heights = params["heights"]
                    msg = self.UpdateDPValue(i, j, dp_array, heights)
                else:
                    msg = "Error: 'i', 'j', 'dp_array' or 'heights' parameter is missing for UPDATE_DP_VALUE action."
                    
            elif action_code == self.FIND_MAX_DP_VALUE:
                if "dp_array" in params:
                    dp_array = params["dp_array"]
                    msg = self.FindMaxDPValue(dp_array)
                else:
                    msg = "Error: 'dp_array' parameter is missing for FIND_MAX_DP_VALUE action."
                    
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
    
        Initialize a DP array of length 'length' with all elements initialized to 1.
    
        Args:
            length (int): The length of the DP array.
    
        Returns:
            str: The initialized DP array, converted to a string using json.dumps.
    
        Example Output:
            "[1, 1, 1, 1]"
        """
        dp_array = [1] * length
        return json.dumps(dp_array)

    def UpdateDPValue(self, i: int, j: int, dp_array: list, heights: list):
        r"""
    
        Update the value at index i in the DP array based on the height relationship between positions i and j.
    
        Args:
            i (int): The current index in the DP array to be updated.
            j (int): The preceding index used for comparison.
            dp_array (list[int]): The current DP array.
            heights (list[int]): The array of students' heights.
    
        Returns:
            str: The updated DP array, converted to a string using json.dumps.
    
        Example Output:
            "[1, 1, 2, 3, 3, 4]"
        """
        if heights[i] >= heights[j]:
            dp_array[i] = max(dp_array[i], dp_array[j] + 1)
        return json.dumps(dp_array)

    def FindMaxDPValue(self, dp_array: list):
        r"""
    
        Find the maximum value in the DP array.
    
        Args:
            dp_array (list[int]): The DP array.
    
        Returns:
            str: The maximum value in the DP array.
    
        Example Output:
            "4"
        """
        return str(max(dp_array))

    def Observe(self):
        r"""
    
        Obtain the array of students' heights in the current environment.
    
        Args:
            None
    
        Returns:
            str: The array of students' heights, converted to a string using json.dumps.
    
        Example Output:
            "[5, 3, 4, 8, 6, 7]"
        """
        return json.dumps(self.heights)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the length of the longest non-decreasing subsequence.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 4, Reference answer: 4, Result: Correct, reward=1"
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
        heights_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        heights = json.loads(heights_str)
        n = len(heights)
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        dp_str = self.step(json.dumps({'name': 'InitializeDPArray', 'parameters': {'length': n}}))[1]
        dp_array = json.loads(dp_str)
        
        for i in range(1, n):
            for j in range(i):
                dp_str = self.step(json.dumps({
                    'name': 'UpdateDPValue',
                    'parameters': {
                        'i': i,
                        'j': j,
                        'dp_array': dp_array,
                        'heights': heights
                    }
                }))[1]
                dp_array = json.loads(dp_str)
        
        max_length = int(self.step(json.dumps({
            'name': 'FindMaxDPValue',
            'parameters': {'dp_array': dp_array}
        }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1: example from problem statement
    print("Test Case 1:")
    test_heights = [5, 3, 4, 8, 6, 7]
    env = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"heights\": {test_heights}}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2: random heights
    print("\nTest Case 2:")
    test_heights = [random.randint(1, 100) for _ in range(random.randint(5, 15))]
    env = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"heights\": {test_heights}}}"
    )
    print(f"Height array: {test_heights}")
    print(env.solve())
    print("step count:", env.step_count)