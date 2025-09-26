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

class SmallestSubarrayEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.EXPAND_WINDOW = 1
        self.SHRINK_WINDOW = 2
        self.UPDATE_MIN_LENGTH = 3
        self.CHECK_COMPLETION = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "ExpandWindow": self.EXPAND_WINDOW,
            "ShrinkWindow": self.SHRINK_WINDOW,
            "UpdateMinLength": self.UPDATE_MIN_LENGTH,
            "CheckCompletion": self.CHECK_COMPLETION,
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
        prefix = "SmallestSubarrayEnv@"
        if not env_str.startswith(prefix):
            return None
        return SmallestSubarrayEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.k = options.get("k", 0)
        self.arr = options.get("arr", [])
        self.current_sum = 0
        self.start = 0
        self.end = 0
        self.min_length = self.n + 1  # Initialize with a value larger than possible maximum
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        min_length = self.n + 1
        current_sum = 0
        start = 0
        
        for end in range(self.n):
            current_sum += self.arr[end]
            
            while current_sum >= self.k:
                min_length = min(min_length, end - start + 1)
                current_sum -= self.arr[start]
                start += 1
                
        return min_length if min_length <= self.n else -1

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
            
            elif action_code == self.EXPAND_WINDOW:
                if "end" in params and "current_sum" in params:
                    end = params["end"]
                    current_sum = params["current_sum"]
                    msg = self.ExpandWindow(end, current_sum)
                else:
                    msg = "Error: 'end' or 'current_sum' parameter is missing for EXPAND_WINDOW action."
                    
            elif action_code == self.SHRINK_WINDOW:
                if "start" in params and "current_sum" in params:
                    start = params["start"]
                    current_sum = params["current_sum"]
                    msg = self.ShrinkWindow(start, current_sum)
                else:
                    msg = "Error: 'start' or 'current_sum' parameter is missing for SHRINK_WINDOW action."
                    
            elif action_code == self.UPDATE_MIN_LENGTH:
                if "min_length" in params and "end" in params and "start" in params:
                    min_length = params["min_length"]
                    end = params["end"]
                    start = params["start"]
                    msg = self.UpdateMinLength(min_length, end, start)
                else:
                    msg = "Error: 'min_length', 'end' or 'start' parameter is missing for UPDATE_MIN_LENGTH action."
                    
            elif action_code == self.CHECK_COMPLETION:
                if "end" in params:
                    end = params["end"]
                    msg = self.CheckCompletion(end)
                else:
                    msg = "Error: 'end' parameter is missing for CHECK_COMPLETION action."
                    
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
    
        Obtain the array in the current environment and the target value k.
    
        Args:
            None
    
        Returns:
            str: Information containing the array and the value of k.
    
        Example Output:
            "{\"n\": 10, \"k\": 15, \"arr\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}"
        """
        observation = {
            "n": self.n,
            "k": self.k,
            "arr": self.arr
        }
        return json.dumps(observation)

    def ExpandWindow(self, end: int, current_sum: int):
        r"""
    
        Expand the window by adding the element at the end position to the current sum.
    
        Args:
            end (int): The end index of the current window.
            current_sum (int): The sum of the current window.
    
        Returns:
            str: The expanded end index and the current sum.
    
        Example Output:
            "{\"end\": 7, \"current_sum\": 36}"
        """
        if end < self.n:
            new_sum = current_sum + self.arr[end]
            new_end = end + 1
            return json.dumps({"end": new_end, "current_sum": new_sum})
        return json.dumps({"end": end, "current_sum": current_sum})

    def ShrinkWindow(self, start: int, current_sum: int):
        r"""
    
        Shrink the window by removing the element at the start position from the current sum.
    
        Args:
            start (int): The start index of the current window.
            current_sum (int): The sum of the current window.
    
        Returns:
            str: The shrunk start index and the current sum.
    
        Example Output:
            "{\"start\": 5, \"current_sum\": 27}"
        """
        new_sum = current_sum - self.arr[start]
        new_start = start + 1
        return json.dumps({"start": new_start, "current_sum": new_sum})

    def UpdateMinLength(self, min_length: int, end: int, start: int):
        r"""
    
        Update the minimum subarray length.
    
        Args:
            min_length (int): The current minimum length.
            end (int): The end index of the current window.
            start (int): The start index of the current window.
    
        Returns:
            str: The updated minimum length.
    
        Example Output:
            "2"
        """
        window_length = end - start
        new_min_length = min(min_length, window_length)
        return str(new_min_length)

    def CheckCompletion(self, end: int):
        r"""
    
        Check if all elements have been processed.
    
        Args:
            end (int): The end index of the current window.
    
        Returns:
            str: A flag indicating whether processing is complete.
    
        Example Output:
            "true"
        """
        return str(end >= self.n).lower()

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        import json
        
        obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        obs_dict = json.loads(obs)
        n = obs_dict['n']
        k = obs_dict['k']
        
        min_length = float('inf')
        start = 0
        current_sum = 0
        end = 0
        
        while True:
            while True:
                completion = self.step(json.dumps({'name': 'CheckCompletion', 'parameters': {'end': end}}))[1]
                if completion == 'true':
                    break
                expand_result = self.step(json.dumps({
                    'name': 'ExpandWindow',
                    'parameters': {'end': end, 'current_sum': current_sum}
                }))[1]
                expand_dict = json.loads(expand_result)
                end = expand_dict['end']
                current_sum = expand_dict['current_sum']
                if current_sum >= k:
                    break
            
            completion = self.step(json.dumps({'name': 'CheckCompletion', 'parameters': {'end': end}}))[1]
            if completion == 'true' and current_sum < k:
                break
            
            while current_sum >= k:
                new_min = self.step(json.dumps({
                    'name': 'UpdateMinLength',
                    'parameters': {'min_length': min_length, 'end': end, 'start': start}
                }))[1]
                min_length = int(new_min)
                shrink_result = self.step(json.dumps({
                    'name': 'ShrinkWindow',
                    'parameters': {'start': start, 'current_sum': current_sum}
                }))[1]
                shrink_dict = json.loads(shrink_result)
                start = shrink_dict['start']
                current_sum = shrink_dict['current_sum']
            
            if completion == 'true':
                break
        
        answer = min_length if min_length != float('inf') else -1
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - example from problem statement
    print("Test Case 1:")
    test_env_str1 = "SmallestSubarrayEnv@{\"n\": 10, \"k\": 15, \"arr\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}"
    env1 = SmallestSubarrayEnv.from_env_str(test_env_str1)
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2 - random case
    print("\nTest Case 2:")
    n2 = 15
    k2 = random.randint(20, 100)
    arr2 = [random.randint(1, 20) for _ in range(n2)]
    test_env_str2 = f"SmallestSubarrayEnv@{{\"n\": {n2}, \"k\": {k2}, \"arr\": {arr2}}}"
    env2 = SmallestSubarrayEnv.from_env_str(test_env_str2)
    print(env2.solve())
    print("step count:", env2.step_count)