# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast
import random

class HouseRobberEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_HOUSE_VALUE = 1
        self.COMPUTE_MAX_VALUE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetHouseValue": self.GET_HOUSE_VALUE,
            "ComputeMaxValue": self.COMPUTE_MAX_VALUE,
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
        prefix = "HouseRobberEnv@"
        if not env_str.startswith(prefix):
            return None
        return HouseRobberEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.houses = options.get("houses", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.houses:
            return 0
        if len(self.houses) == 1:
            return self.houses[0]
        
        prev1, prev2 = 0, 0
        for amount in self.houses:
            temp = prev1
            prev1 = max(prev2 + amount, prev1)
            prev2 = temp
        
        return prev1

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
            
            elif action_code == self.GET_HOUSE_VALUE:
                if "index" in params:
                    index = params["index"]
                    msg = self.GetHouseValue(index)
                else:
                    msg = "Error: 'index' parameter is missing for GET_HOUSE_VALUE action."
            
            elif action_code == self.COMPUTE_MAX_VALUE:
                if "prev1" in params and "prev2" in params and "amount" in params:
                    prev1 = params["prev1"]
                    prev2 = params["prev2"]
                    amount = params["amount"]
                    msg = self.ComputeMaxValue(prev1, prev2, amount)
                else:
                    msg = "Error: 'prev1', 'prev2' or 'amount' parameter is missing for COMPUTE_MAX_VALUE action."
                    
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
    
        Obtain the list of cash amounts of all current houses.
    
        Args:
            None
    
        Returns:
            str: A string representation of the list of house cash amounts.
    
        Example Output:
            "[1, 2, 3, 1]"
        """
        return json.dumps(self.houses)

    def GetHouseValue(self, index: int):
        r"""
    
        Obtain the cash amount of the house at the specified index position.
    
        Args:
            index (int): The index position of the house.
    
        Returns:
            str: The cash amount of the house at the specified index position.
    
        Example Output:
            "3"
        """
        if 0 <= index < len(self.houses):
            return str(self.houses[index])
        return "Error: Invalid house index"

    def ComputeMaxValue(self, prev1: int, prev2: int, amount: int):
        r"""
    
        Calculate the new maximum robbery amount based on the maximum robbery amounts of the previous two states and the current house amount.
    
        Args:
            prev1 (int): The maximum robbery amount of the previous state.
            prev2 (int): The maximum robbery amount of the state two steps before.
            amount (int): The cash amount of the current house.
    
        Returns:
            str: A JSON string containing the new prev1 and prev2 values.
    
        Example Output:
            "{\"new_prev1\": 4, \"new_prev2\": 2}"
        """
        temp = prev1
        new_prev1 = max(prev2 + amount, prev1)
        new_prev2 = temp
        return json.dumps({"new_prev1": new_prev1, "new_prev2": new_prev2})

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The user-submitted answer for the maximum robbery amount.
    
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
        houses_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        houses = ast.literal_eval(houses_str)
        n = len(houses)
        
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        if n == 1:
            max_value = int(self.step(json.dumps({'name': 'GetHouseValue', 'parameters': {'index': 0}}))[1])
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_value}}))[1]
        
        prev2 = 0
        prev1 = int(self.step(json.dumps({'name': 'GetHouseValue', 'parameters': {'index': 0}}))[1])
        
        if n == 2:
            current = int(self.step(json.dumps({'name': 'GetHouseValue', 'parameters': {'index': 1}}))[1])
            max_value = max(prev1, current)
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_value}}))[1]
        
        for i in range(1, n):
            current_amount = int(self.step(json.dumps({'name': 'GetHouseValue', 'parameters': {'index': i}}))[1])
            compute_result = self.step(json.dumps({
                'name': 'ComputeMaxValue',
                'parameters': {'prev1': prev1, 'prev2': prev2, 'amount': current_amount}
            }))[1]
            compute_dict = json.loads(compute_result)
            prev1, prev2 = compute_dict['new_prev1'], compute_dict['new_prev2']
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': prev1}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_houses1 = [1, 2, 3, 1]
    env1 = HouseRobberEnv.from_env_str(f"HouseRobberEnv@{{\"houses\": {test_houses1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_houses2 = [random.randint(10, 100) for _ in range(6)]
    env2 = HouseRobberEnv.from_env_str(f"HouseRobberEnv@{{\"houses\": {test_houses2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)