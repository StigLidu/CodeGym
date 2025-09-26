# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class UniqueBSTCountEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.SET_BASE_CASES = 0
        self.CALCULATE_DP = 1
        self.GET_FINAL_ANSWER = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "SetBaseCases": self.SET_BASE_CASES,
            "CalculateDP": self.CALCULATE_DP,
            "GetFinalAnswer": self.GET_FINAL_ANSWER,
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
        prefix = "UniqueBSTCountEnv@"
        if not env_str.startswith(prefix):
            return None
        return UniqueBSTCountEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 1)
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
        if self.n == 0 or self.n == 1:
            return 1
        
        dp = [0] * (self.n + 1)
        dp[0] = 1
        dp[1] = 1
        
        for i in range(2, self.n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        
        return dp[self.n]

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
            
            if action_code == self.SET_BASE_CASES:
                if "size" in params:
                    size = params["size"]
                    msg = self.SetBaseCases(size)
                else:
                    msg = "Error: 'size' parameter is missing for SET_BASE_CASES action."
            
            elif action_code == self.CALCULATE_DP:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.CalculateDP(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for CALCULATE_DP action."
                    
            elif action_code == self.GET_FINAL_ANSWER:
                if "i" in params:
                    i = params["i"]
                    msg = self.GetFinalAnswer(i)
                else:
                    msg = "Error: 'i' parameter is missing for GET_FINAL_ANSWER action."
                    
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
    def SetBaseCases(self, size: int):
        r"""
    
        Initialize the DP array and set the base cases dp[0] = 1 and dp[1] = 1.
    
        Args:
            size (int): The size of the DP array, which should be set to n+1.
    
        Returns:
            str: The initialized DP array.
    
        Example Output:
            "[1, 1, 0, 0]"
        """
        self.dp = [0] * size
        self.dp[0] = 1
        self.dp[1] = 1
        return str(self.dp)

    def CalculateDP(self, i: int, j: int):
        r"""
    
        Calculate the value of dp[i] according to the formula dp[i] += dp[j-1] * dp[i-j].
    
        Args:
            i (int): The current DP index to be calculated.
            j (int): The currently selected root node position.
    
        Returns:
            str: The updated value of dp[i].
    
        Example Output:
            "5"
        """
        if i >= len(self.dp) or j > i:
            return "Error: i or j is out of bounds"
        
        self.dp[i] += self.dp[j - 1] * self.dp[i - j]
        return str(self.dp[i])

    def GetFinalAnswer(self, i: int):
        r"""
    
        Get the value of dp[i], which is the final answer.
    
        Args:
            i (int): The DP index to be retrieved, which should be set to n.
    
        Returns:
            str: The value of dp[i].
    
        Example Output:
            "5"
        """
        if i >= len(self.dp):
            return "Error: i is out of bounds"
        return str(self.dp[i])

    def Observe(self):
        r"""
    
        Return the observation information of the current state, including the value of n and the current dp array.
    
        Args:
            None
    
        Returns:
            str: Information describing the current state.
    
        Example Output:
            "n=3, dp=[1, 1, 2, 0]"
        """
        return f"n={self.n}, dp={self.dp}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 5, Reference answer: 5, Result: Correct, reward=1"
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
        obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = int(obs.split('n=')[1].split(',')[0])
        
        self.step(json.dumps({'name': 'SetBaseCases', 'parameters': {'size': n + 1}}))
        
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                self.step(json.dumps({'name': 'CalculateDP', 'parameters': {'i': i, 'j': j}}))
        
        final_answer = int(self.step(json.dumps({'name': 'GetFinalAnswer', 'parameters': {'i': n}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (example from problem statement)
    print("Test Case 1:")
    env = UniqueBSTCountEnv.from_env_str("UniqueBSTCountEnv@{\"n\": 3}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = UniqueBSTCountEnv.from_env_str("UniqueBSTCountEnv@{\"n\": 5}")
    print(env.solve())
    print("step count:", env.step_count)