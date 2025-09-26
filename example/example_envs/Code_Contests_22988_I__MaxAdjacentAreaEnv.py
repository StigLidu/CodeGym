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

class MaxAdjacentAreaEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_AREA = 0
        self.UPDATE_MAX_AREA = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateArea": self.CALCULATE_AREA,
            "UpdateMaxArea": self.UPDATE_MAX_AREA,
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
        prefix = "MaxAdjacentAreaEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxAdjacentAreaEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
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
        max_area = 0
        for i in range(self.n - 1):
            area = min(self.heights[i], self.heights[i + 1]) * 1
            max_area = max(max_area, area)
        return max_area

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
            
            if action_code == self.CALCULATE_AREA:
                if "index" in params:
                    index = params["index"]
                    msg = self.CalculateArea(index)
                else:
                    msg = "Error: 'index' parameter is missing for CALCULATE_AREA action."
            
            elif action_code == self.UPDATE_MAX_AREA:
                if "current_area" in params and "current_max" in params:
                    current_area = params["current_area"]
                    current_max = params["current_max"]
                    msg = self.UpdateMaxArea(current_area, current_max)
                else:
                    msg = "Error: 'current_area' or 'current_max' parameter is missing for UPDATE_MAX_AREA action."
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
    def CalculateArea(self, index: int):
        r"""
    
        Calculate the area of the rectangle formed by the index-th and (index+1)-th line segments.
    
        Args:
            index (int): The index of the line segment.
    
        Returns:
            str: The area of the formed rectangle.
    
        Example Output:
            "5"
        """
        if index < 0 or index >= self.n - 1:
            return "Error: index out of range"
        area = min(self.heights[index], self.heights[index + 1]) * 1
        return str(area)

    def UpdateMaxArea(self, current_area: int, current_max: int):
        r"""
    
        Compare the current area with the current maximum value and return the updated maximum value.
    
        Args:
            current_area (int): The currently calculated area.
            current_max (int): The current maximum area.
    
        Returns:
            str: The updated maximum area.
    
        Example Output:
            "6"
        """
        new_max = max(current_max, current_area)
        return str(new_max)

    def Observe(self):
        r"""
    
        Return the number of line segments and their height information.
    
        Args:
            None
    
        Returns:
            str: The number of line segments and the list of heights.
    
        Example Output:
            "n=6, heights=[2, 1, 5, 6, 2, 3]"
        """
        return f"n={self.n}, heights={self.heights}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n_str = observe_result.split(',')[0].strip()
        n = int(n_str.split('=')[1])
        current_max = 0
        for index in range(n - 1):
            area_str = self.step(json.dumps({'name': 'CalculateArea', 'parameters': {'index': index}}))[1]
            current_area = int(area_str)
            new_max_str = self.step(json.dumps({'name': 'UpdateMaxArea', 'parameters': {'current_area': current_area, 'current_max': current_max}}))[1]
            current_max = int(new_max_str)
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': current_max}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (from examples)
    print("Test Case 1:")
    test_n1 = 6
    test_heights1 = [2, 1, 5, 6, 2, 3]
    env1 = MaxAdjacentAreaEnv.from_env_str(f"MaxAdjacentAreaEnv@{{\"n\": {test_n1}, \"heights\": {test_heights1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2 (random)
    print("\nTest Case 2:")
    test_n2 = random.randint(2, 10)
    test_heights2 = [random.randint(1, 10) for _ in range(test_n2)]
    env2 = MaxAdjacentAreaEnv.from_env_str(f"MaxAdjacentAreaEnv@{{\"n\": {test_n2}, \"heights\": {test_heights2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)