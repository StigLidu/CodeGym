# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestSubarrayEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_SUBARRAY = 0
        self.CALCULATE_LENGTH = 1
        self.UPDATE_MAX_LENGTH = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckSubarray": self.CHECK_SUBARRAY,
            "CalculateLength": self.CALCULATE_LENGTH,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
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
        prefix = "LongestSubarrayEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestSubarrayEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.nums = options.get("nums", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.nums)
        max_len = 0
        
        for i in range(n):
            has_positive = has_negative = has_zero = False
            for j in range(i, n):
                if self.nums[j] > 0:
                    has_positive = True
                elif self.nums[j] < 0:
                    has_negative = True
                elif self.nums[j] == 0:
                    has_zero = True
                
                if has_positive and has_negative and has_zero:
                    max_len = max(max_len, j - i + 1)
        
        return max_len

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
            
            if action_code == self.CHECK_SUBARRAY:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CheckSubarray(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for CHECK_SUBARRAY action."
            
            elif action_code == self.CALCULATE_LENGTH:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CalculateLength(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for CALCULATE_LENGTH action."
                    
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "current_max" in params and "candidate" in params:
                    current_max = params["current_max"]
                    candidate = params["candidate"]
                    msg = self.UpdateMaxLength(current_max, candidate)
                else:
                    msg = "Error: 'current_max' or 'candidate' parameter is missing for UPDATE_MAX_LENGTH action."
                    
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
    def CheckSubarray(self, start: int, end: int):
        r"""
    
        Check if the subarray from start to end (inclusive) contains at least one positive number, one negative number, and one zero.
    
        Args:
            start (int): Starting index of the subarray.
            end (int): Ending index of the subarray.
    
        Returns:
            str: "True" indicates the subarray meets the condition, "False" indicates it does not.
    
        Example Output:
            "True"
        """
        has_positive = has_negative = has_zero = False
        for i in range(start, end + 1):
            if self.nums[i] > 0:
                has_positive = True
            elif self.nums[i] < 0:
                has_negative = True
            elif self.nums[i] == 0:
                has_zero = True
                
        return str(has_positive and has_negative and has_zero)

    def CalculateLength(self, start: int, end: int):
        r"""
    
        Calculate the length of the subarray from start to end (inclusive).
    
        Args:
            start (int): Starting index of the subarray.
            end (int): Ending index of the subarray.
    
        Returns:
            str: String representation of the subarray length.
    
        Example Output:
            "3"
        """
        return str(end - start + 1)

    def UpdateMaxLength(self, current_max: int, candidate: int):
        r"""
    
        Compare the current maximum value with the candidate value and return the larger one.
    
        Args:
            current_max (int): Current maximum value.
            candidate (int): Candidate value.
    
        Returns:
            str: String representation of the new maximum value.
    
        Example Output:
            "5"
        """
        return str(max(current_max, candidate))

    def Observe(self):
        r"""
    
        Return the list of integers in the current environment.
    
        Args:
            None
    
        Returns:
            str: String representation of the list of integers in the environment.
    
        Example Output:
            "[1, -2, 0, 3, 4]"
        """
        return str(self.nums)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        nums_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        nums = ast.literal_eval(nums_str)
        n = len(nums)
        max_len = 0
        
        for start in range(n):
            for end in range(start, n):
                check_result = self.step(json.dumps({
                    'name': 'CheckSubarray',
                    'parameters': {'start': start, 'end': end}
                }))[1]
                if check_result == "True":
                    current_len_str = self.step(json.dumps({
                        'name': 'CalculateLength',
                        'parameters': {'start': start, 'end': end}
                    }))[1]
                    current_len = int(current_len_str)
                    max_len_str = self.step(json.dumps({
                        'name': 'UpdateMaxLength',
                        'parameters': {'current_max': max_len, 'candidate': current_len}
                    }))[1]
                    max_len = int(max_len_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_len}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_nums1 = [1, -2, 0, 3, 4, -1, 5, 0]
    env1 = LongestSubarrayEnv.from_env_str(f"LongestSubarrayEnv@{{\"nums\": {test_nums1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_nums2 = [1, 2, 3, -1, -2]
    env2 = LongestSubarrayEnv.from_env_str(f"LongestSubarrayEnv@{{\"nums\": {test_nums2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)