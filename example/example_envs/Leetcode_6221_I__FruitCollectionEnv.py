# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class FruitCollectionEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.SET_LEFT_POINTER = 1
        self.SET_RIGHT_POINTER = 2
        self.COUNT_FRUIT_TYPES = 3
        self.CALCULATE_WINDOW_SIZE = 4
        self.UPDATE_MAX_FRUITS = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "SetLeftPointer": self.SET_LEFT_POINTER,
            "SetRightPointer": self.SET_RIGHT_POINTER,
            "CountFruitTypes": self.COUNT_FRUIT_TYPES,
            "CalculateWindowSize": self.CALCULATE_WINDOW_SIZE,
            "UpdateMaxFruits": self.UPDATE_MAX_FRUITS,
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
        prefix = "FruitCollectionEnv@"
        if not env_str.startswith(prefix):
            return None
        return FruitCollectionEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.fruits = options.get("fruits", [])
        self.left = 0
        self.right = 0
        self.max_fruits = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        basket = {}
        left = 0
        max_fruits = 0
        
        for right in range(len(self.fruits)):
            if self.fruits[right] in basket:
                basket[self.fruits[right]] += 1
            else:
                basket[self.fruits[right]] = 1

            while len(basket) > 2:
                basket[self.fruits[left]] -= 1
                if basket[self.fruits[left]] == 0:
                    del basket[self.fruits[left]]
                left += 1
            
            max_fruits = max(max_fruits, right - left + 1)
        
        return max_fruits

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
            
            elif action_code == self.SET_LEFT_POINTER:
                if "position" in params:
                    position = params["position"]
                    msg = self.SetLeftPointer(position)
                else:
                    msg = "Error: 'position' parameter is missing for SET_LEFT_POINTER action."
            
            elif action_code == self.SET_RIGHT_POINTER:
                if "position" in params:
                    position = params["position"]
                    msg = self.SetRightPointer(position)
                else:
                    msg = "Error: 'position' parameter is missing for SET_RIGHT_POINTER action."
            
            elif action_code == self.COUNT_FRUIT_TYPES:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CountFruitTypes(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for COUNT_FRUIT_TYPES action."
            
            elif action_code == self.CALCULATE_WINDOW_SIZE:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateWindowSize(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_WINDOW_SIZE action."
            
            elif action_code == self.UPDATE_MAX_FRUITS:
                if "current" in params and "max_so_far" in params:
                    current = params["current"]
                    max_so_far = params["max_so_far"]
                    msg = self.UpdateMaxFruits(current, max_so_far)
                else:
                    msg = "Error: 'current' or 'max_so_far' parameter is missing for UPDATE_MAX_FRUITS action."
            
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
    
        Obtain the information of the current fruit array.
    
        Args:
            None
    
        Returns:
            str: The string representation of the current fruit array.
    
        Example Output:
            "[1, 2, 1, 3, 4, 3]"
        """
        return str(self.fruits)

    def SetLeftPointer(self, position: int):
        r"""
    
        Set the position of the left pointer.
    
        Args:
            position (int): The position to set the left pointer to.
    
        Returns:
            str: The position of the left pointer after setting.
    
        Example Output:
            "1"
        """
        self.left = position
        return str(self.left)

    def SetRightPointer(self, position: int):
        r"""
    
        Set the position of the right pointer.
    
        Args:
            position (int): The position to set the right pointer to.
    
        Returns:
            str: The position of the right pointer after setting.
    
        Example Output:
            "3"
        """
        self.right = position
        return str(self.right)

    def CountFruitTypes(self, left: int, right: int):
        r"""
    
        Calculate the number of fruit types within the current window.
    
        Args:
            left (int): The left boundary index of the window.
            right (int): The right boundary index of the window.
    
        Returns:
            str: The number of fruit types in the window.
    
        Example Output:
            "2"
        """
        if left < 0 or right >= len(self.fruits) or left > right:
            return "0"
        fruit_types = set(self.fruits[left:right+1])
        return str(len(fruit_types))

    def CalculateWindowSize(self, left: int, right: int):
        r"""
    
        Calculate the size of the current window.
    
        Args:
            left (int): The left boundary index of the window.
            right (int): The right boundary index of the window.
    
        Returns:
            str: The size of the window.
    
        Example Output:
            "3"
        """
        if left < 0 or right >= len(self.fruits) or left > right:
            return "0"
        return str(right - left + 1)

    def UpdateMaxFruits(self, current: int, max_so_far: int):
        r"""
    
        Update the maximum number of fruits.
    
        Args:
            current (int): The number of fruits in the current window.
            max_so_far (int): The maximum number of fruits so far.
    
        Returns:
            str: The updated maximum number of fruits.
    
        Example Output:
            "5"
        """
        self.max_fruits = max(current, max_so_far)
        return str(self.max_fruits)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
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
        fruits_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        fruits = ast.literal_eval(fruits_str)
        n = len(fruits)
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        max_fruits = 0
        left = 0
        
        for right in range(n):
            self.step(json.dumps({'name': 'SetLeftPointer', 'parameters': {'position': left}}))
            self.step(json.dumps({'name': 'SetRightPointer', 'parameters': {'position': right}}))
            
            type_count = int(self.step(json.dumps({'name': 'CountFruitTypes', 'parameters': {'left': left, 'right': right}}))[1])
            
            while type_count > 2:
                left += 1
                self.step(json.dumps({'name': 'SetLeftPointer', 'parameters': {'position': left}}))
                type_count = int(self.step(json.dumps({'name': 'CountFruitTypes', 'parameters': {'left': left, 'right': right}}))[1])
            
            current_size = int(self.step(json.dumps({'name': 'CalculateWindowSize', 'parameters': {'left': left, 'right': right}}))[1])
            
            max_fruits = int(self.step(json.dumps({'name': 'UpdateMaxFruits', 'parameters': {'current': current_size, 'max_so_far': max_fruits}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_fruits}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_fruits1 = [1, 2, 1, 3, 4, 3, 2, 2]
    env1 = FruitCollectionEnv.from_env_str(f"FruitCollectionEnv@{{\"fruits\": {test_fruits1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_fruits2 = [3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4]
    env2 = FruitCollectionEnv.from_env_str(f"FruitCollectionEnv@{{\"fruits\": {test_fruits2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)