# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MaxBonusEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_INITIAL_WINDOW_SUM = 0
        self.SLIDE_WINDOW = 1
        self.COMPARE_SUMS = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateInitialWindowSum": self.CALCULATE_INITIAL_WINDOW_SUM,
            "SlideWindow": self.SLIDE_WINDOW,
            "CompareSums": self.COMPARE_SUMS,
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
        prefix = "MaxBonusEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxBonusEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.k = options.get("k", 0)
        self.scores = options.get("scores", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if self.k == 0 or self.n == 0:
            return 0
            
        current_sum = sum(self.scores[:self.k])
        max_sum = current_sum
        
        for i in range(self.k, self.n):
            current_sum += self.scores[i] - self.scores[i - self.k]
            if current_sum > max_sum:
                max_sum = current_sum
                
        return max_sum

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
            
            if action_code == self.CALCULATE_INITIAL_WINDOW_SUM:
                if "end_index" in params:
                    end_index = params["end_index"]
                    msg = self.CalculateInitialWindowSum(end_index)
                else:
                    msg = "Error: 'end_index' parameter is missing for CALCULATE_INITIAL_WINDOW_SUM action."
            
            elif action_code == self.SLIDE_WINDOW:
                if "current_sum" in params and "i" in params and "k" in params:
                    current_sum = params["current_sum"]
                    i = params["i"]
                    k_val = params["k"]
                    msg = self.SlideWindow(current_sum, i, k_val)
                else:
                    msg = "Error: 'current_sum', 'i' or 'k' parameter is missing for SLIDE_WINDOW action."
                    
            elif action_code == self.COMPARE_SUMS:
                if "sum1" in params and "sum2" in params:
                    sum1 = params["sum1"]
                    sum2 = params["sum2"]
                    msg = self.CompareSums(sum1, sum2)
                else:
                    msg = "Error: 'sum1' or 'sum2' parameter is missing for COMPARE_SUMS action."
                    
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
    def CalculateInitialWindowSum(self, end_index: int):
        r"""
    
        Calculate the total score from the start up to (but not including) end_index.
    
        Args:
            end_index (int): The ending index of the window (exclusive).
    
        Returns:
            str: The total score within the window.
    
        Example Output:
            "5"
        """
        window_sum = sum(self.scores[:end_index])
        return str(window_sum)

    def SlideWindow(self, current_sum: int, i: int, k: int):
        r"""
    
        Slide the window to calculate the total sum of the new window.
    
        Args:
            current_sum (int): The total sum of the current window.
            i (int): The index of the ending position of the new window.
            k (int): The size of the window.
    
        Returns:
            str: The total sum of the new window after sliding.
    
        Example Output:
            "7"
        """
        new_sum = current_sum + self.scores[i] - self.scores[i - k]
        return str(new_sum)

    def CompareSums(self, sum1: int, sum2: int):
        r"""
    
        Compare two totals and return the larger one.
    
        Args:
            sum1 (int): The first total to be compared.
            sum2 (int): The second total to be compared.
    
        Returns:
            str: The larger total.
    
        Example Output:
            "7"
        """
        return str(max(sum1, sum2))

    def Observe(self):
        r"""
    
        Obtain the number of days n, window size k, and the score list in the current environment.
    
        Args:
            None
    
        Returns:
            str: Information containing n, k, and scores.
    
        Example Output:
            "n=6, k=2, scores=[2, 1, 3, -1, 4, 2]"
        """
        return f"n={self.n}, k={self.k}, scores={self.scores}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
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
        observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        parts = observe_info.split(', ')
        n = int(parts[0].split('=')[1])
        k = int(parts[1].split('=')[1])
        
        initial_sum = int(self.step(json.dumps({'name': 'CalculateInitialWindowSum', 'parameters': {'end_index': k}}))[1])
        max_sum = initial_sum
        
        for i in range(k, n):
            current_sum = int(self.step(json.dumps({'name': 'SlideWindow', 'parameters': {'current_sum': max_sum if i == k else current_sum, 'i': i, 'k': k}}))[1])
            max_sum = int(self.step(json.dumps({'name': 'CompareSums', 'parameters': {'sum1': max_sum, 'sum2': current_sum}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_sum}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_env_str1 = "MaxBonusEnv@{\"n\": 6, \"k\": 2, \"scores\": [2, 1, 3, -1, 4, 2]}"
    env1 = MaxBonusEnv.from_env_str(test_env_str1)
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_env_str2 = "MaxBonusEnv@{\"n\": 7, \"k\": 3, \"scores\": [-1, -2, -3, -4, -5, -6, -7]}"
    env2 = MaxBonusEnv.from_env_str(test_env_str2)
    print(env2.solve())
    print("step count:", env2.step_count)