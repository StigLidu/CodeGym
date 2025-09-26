# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class LongestIncreasingSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_SUBSEQUENCE = 0
        self.BINARY_SEARCH_POSITION = 1
        self.UPDATE_SUBSEQUENCE = 2
        self.GET_SUBSEQUENCE_LENGTH = 3
        self.OBSERVE = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeSubsequence": self.INITIALIZE_SUBSEQUENCE,
            "BinarySearchPosition": self.BINARY_SEARCH_POSITION,
            "UpdateSubsequence": self.UPDATE_SUBSEQUENCE,
            "GetSubsequenceLength": self.GET_SUBSEQUENCE_LENGTH,
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
        self.nums = options.get("nums", [])
        self.subsequence = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.nums:
            return 0

        sub = []
        for num in self.nums:
            left, right = 0, len(sub)
            while left < right:
                mid = (left + right) // 2
                if sub[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            if left == len(sub):
                sub.append(num)
            else:
                sub[left] = num
        
        return len(sub)

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
            
            if action_code == self.INITIALIZE_SUBSEQUENCE:
                msg = self.InitializeSubsequence()
            
            elif action_code == self.BINARY_SEARCH_POSITION:
                if "num" in params and "subsequence" in params:
                    num = params["num"]
                    subsequence = params["subsequence"]
                    msg = self.BinarySearchPosition(num, subsequence)
                else:
                    msg = "Error: 'num' or 'subsequence' parameter is missing for BINARY_SEARCH_POSITION action."
            
            elif action_code == self.UPDATE_SUBSEQUENCE:
                if "subsequence" in params and "position" in params and "num" in params:
                    subsequence = params["subsequence"]
                    position = params["position"]
                    num = params["num"]
                    msg = self.UpdateSubsequence(subsequence, position, num)
                else:
                    msg = "Error: 'subsequence', 'position' or 'num' parameter is missing for UPDATE_SUBSEQUENCE action."
            
            elif action_code == self.GET_SUBSEQUENCE_LENGTH:
                if "subsequence" in params:
                    subsequence = params["subsequence"]
                    msg = self.GetSubsequenceLength(subsequence)
                else:
                    msg = "Error: 'subsequence' parameter is missing for GET_SUBSEQUENCE_LENGTH action."
            
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
    def InitializeSubsequence(self):
        r"""
    
        Initialize an empty subsequence.
    
        Args:
            None
    
        Returns:
            str: The string representation of the initialized empty list.
    
        Example Output:
            "[]"
        """
        return "[]"

    def BinarySearchPosition(self, num: int, subsequence: list):
        r"""
    
        Use binary search to determine the insertion position of a number in the subsequence.
    
        Args:
            num (int): The number whose position is to be found.
            subsequence (list[int]): The current subsequence.
    
        Returns:
            str: The found insertion position.
    
        Example Output:
            "2"
        """
        left, right = 0, len(subsequence)
        while left < right:
            mid = (left + right) // 2
            if subsequence[mid] < num:
                left = mid + 1
            else:
                right = mid
        return str(left)

    def UpdateSubsequence(self, subsequence: list, position: int, num: int):
        r"""
    
        Update the subsequence according to the position.
    
        Args:
            subsequence (list[int]): The current subsequence.
            position (int): The found insertion position.
            num (int): The number to be inserted or replaced.
    
        Returns:
            str: The string representation of the updated subsequence.
    
        Example Output:
            "[2, 5, 7]"
        """
        if position == len(subsequence):
            subsequence.append(num)
        else:
            subsequence[position] = num
        return str(subsequence)

    def GetSubsequenceLength(self, subsequence: list):
        r"""
    
        Get the length of the subsequence.
    
        Args:
            subsequence (list[int]): The current subsequence.
    
        Returns:
            str: The length of the subsequence.
    
        Example Output:
            "4"
        """
        return str(len(subsequence))

    def Observe(self):
        r"""
    
        Return the list of integers in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string representation of the list of integers in the current environment.
    
        Example Output:
            "[10, 9, 2, 5, 3, 7, 101, 18]"
        """
        return str(self.nums)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
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
        nums_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        nums = ast.literal_eval(nums_str)
        
        subsequence_str = self.step(json.dumps({'name': 'InitializeSubsequence', 'parameters': {}}))[1]
        subsequence = ast.literal_eval(subsequence_str)
        
        for num in nums:
            pos_str = self.step(json.dumps({
                'name': 'BinarySearchPosition',
                'parameters': {'num': num, 'subsequence': subsequence}
            }))[1]
            pos = int(pos_str)
            
            subsequence_str = self.step(json.dumps({
                'name': 'UpdateSubsequence',
                'parameters': {'subsequence': subsequence, 'position': pos, 'num': num}
            }))[1]
            subsequence = ast.literal_eval(subsequence_str)
        
        length_str = self.step(json.dumps({
            'name': 'GetSubsequenceLength',
            'parameters': {'subsequence': subsequence}
        }))[1]
        length = int(length_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_nums = [10, 9, 2, 5, 3, 7, 101, 18]
    env = LongestIncreasingSubsequenceEnv.from_env_str(f"LongestIncreasingSubsequenceEnv@{{\"nums\": {test_nums}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_nums = [0, 1, 0, 3, 2, 3]
    env = LongestIncreasingSubsequenceEnv.from_env_str(f"LongestIncreasingSubsequenceEnv@{{\"nums\": {test_nums}}}")
    print(env.solve())
    print("step count:", env.step_count)