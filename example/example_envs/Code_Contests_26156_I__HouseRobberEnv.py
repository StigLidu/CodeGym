# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from typing import List

class HouseRobberEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.INITIALIZE_DP = 1
        self.COMPUTE_DP_VALUE = 2
        self.GET_MAX_GOLD = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "InitializeDp": self.INITIALIZE_DP,
            "ComputeDpValue": self.COMPUTE_DP_VALUE,
            "GetMaxGold": self.GET_MAX_GOLD,
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
        prefix = "HouseRobberEnv@"
        if not env_str.startswith(prefix):
            return None
        return HouseRobberEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.gold = options.get("gold", [])
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
        if self.n == 0:
            return 0
        if self.n == 1:
            return self.gold[0]
        
        dp = [0] * self.n
        dp[0] = self.gold[0]
        dp[1] = max(self.gold[0], self.gold[1])
        
        for i in range(2, self.n):
            dp[i] = max(dp[i-1], dp[i-2] + self.gold[i])
        
        return dp[-1]

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
            
            elif action_code == self.INITIALIZE_DP:
                if "n" in params and "gold" in params:
                    n = params["n"]
                    gold = params["gold"]
                    msg = self.InitializeDp(n, gold)
                else:
                    msg = "Error: 'n' or 'gold' parameter is missing for INITIALIZE_DP action."
            
            elif action_code == self.COMPUTE_DP_VALUE:
                if "i" in params and "gold_i" in params and "dp" in params:
                    i = params["i"]
                    gold_i = params["gold_i"]
                    dp = params["dp"]
                    msg = self.ComputeDpValue(i, gold_i, dp)
                else:
                    msg = "Error: 'i', 'gold_i' or 'dp' parameter is missing for COMPUTE_DP_VALUE action."
            
            elif action_code == self.GET_MAX_GOLD:
                if "dp" in params:
                    dp = params["dp"]
                    msg = self.GetMaxGold(dp)
                else:
                    msg = "Error: 'dp' parameter is missing for GET_MAX_GOLD action."
            
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
    
        Obtain the number of houses in the current environment and the amount of gold in each house.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the number of houses and the amount of gold.
    
        Example Output:
            "{\"n\": 3, \"gold\": [100, 1, 100]}"
        """
        observation = {
            "n": self.n,
            "gold": self.gold
        }
        return json.dumps(observation)

    def InitializeDp(self, n: int, gold: List[int]):
        r"""
    
        Initialize the dynamic programming array and set the values of the first two elements.
    
        Args:
            n (int): The number of houses.
            gold (List[int]): The amount of gold in each house.
    
        Returns:
            str: A JSON string of the initialized dynamic programming array.
    
        Example Output:
            "[100, 100]"
        """
        if n == 0:
            self.dp = []
        elif n == 1:
            self.dp = [gold[0]]
        else:
            self.dp = [0] * n
            self.dp[0] = gold[0]
            self.dp[1] = max(gold[0], gold[1])
        return json.dumps(self.dp)

    def ComputeDpValue(self, i: int, gold_i: int, dp: List[int]):
        r"""
    
        Calculate the value at the i-th position in the dynamic programming array.
    
        Args:
            i (int): The index of the position to be calculated.
            gold_i (int): The amount of gold in the i-th house.
            dp (List[int]): The current dynamic programming array.
    
        Returns:
            str: A JSON string of the updated dynamic programming array.
    
        Example Output:
            "[100, 100, 200]"
        """
        if i >= len(dp):
            raise ValueError("Index i is out of bounds for dp array")
            
        dp[i] = max(dp[i-1], dp[i-2] + gold_i)
        self.dp = dp
        return json.dumps(self.dp)

    def GetMaxGold(self, dp: List[int]):
        r"""
    
        Obtain the maximum amount of gold from the dynamic programming array.
    
        Args:
            dp (List[int]): The dynamic programming array.
    
        Returns:
            str: A string representation of the maximum amount of gold.
    
        Example Output:
            "200"
        """
        if not dp:
            return "0"
        return str(dp[-1])

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 200, Reference answer: 200, Result: Correct, reward=1"
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
        observe_data = json.loads(observe_result)
        n = observe_data['n']
        gold = observe_data['gold']
        
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        if n == 1:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': gold[0]}}))[1]
        
        dp_init = self.step(json.dumps({'name': 'InitializeDp', 'parameters': {'n': n, 'gold': gold}}))[1]
        dp = json.loads(dp_init)
        
        for i in range(2, n):
            gold_i = gold[i]
            dp_str = self.step(json.dumps({'name': 'ComputeDpValue', 'parameters': {'i': i, 'gold_i': gold_i, 'dp': dp}}))[1]
            dp = json.loads(dp_str)
        
        max_gold_str = self.step(json.dumps({'name': 'GetMaxGold', 'parameters': {'dp': dp}}))[1]
        max_gold = int(max_gold_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_gold}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_case1 = {"n": 5, "gold": [1, 2, 3, 1, 5]}
    env = HouseRobberEnv.from_env_str(f"HouseRobberEnv@{test_case1}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_case2 = {"n": 3, "gold": [100, 1, 100]}
    env = HouseRobberEnv.from_env_str(f"HouseRobberEnv@{test_case2}")
    print(env.solve())
    print("step count:", env.step_count)