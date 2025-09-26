# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class TeamAssignmentEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_SUM = 0
        self.CHECK_CONDITION = 1
        self.INCREMENT_COUNT = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateSum": self.CALCULATE_SUM,
            "CheckCondition": self.CHECK_CONDITION,
            "IncrementCount": self.INCREMENT_COUNT,
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
        prefix = "TeamAssignmentEnv@"
        if not env_str.startswith(prefix):
            return None
        return TeamAssignmentEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.N = options.get("N", 0)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if self.N == 0:
            return 0
        
        count = 0
        sum_so_far = 0
        
        while sum_so_far + (count + 1) <= self.N:
            count += 1
            sum_so_far += count
        
        return count

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
            
            if action_code == self.CALCULATE_SUM:
                if "count" in params and "sum_so_far" in params:
                    count = params["count"]
                    sum_so_far = params["sum_so_far"]
                    msg = self.CalculateSum(count, sum_so_far)
                else:
                    msg = "Error: 'count' or 'sum_so_far' parameter is missing for CALCULATE_SUM action."
            
            elif action_code == self.CHECK_CONDITION:
                if "sum_so_far" in params and "count" in params:
                    sum_so_far = params["sum_so_far"]
                    count = params["count"]
                    msg = self.CheckCondition(sum_so_far, count)
                else:
                    msg = "Error: 'sum_so_far' or 'count' parameter is missing for CHECK_CONDITION action."
            
            elif action_code == self.INCREMENT_COUNT:
                if "count" in params:
                    count = params["count"]
                    msg = self.IncrementCount(count)
                else:
                    msg = "Error: 'count' parameter is missing for INCREMENT_COUNT action."
            
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
    def CalculateSum(self, count: int, sum_so_far: int):
        r"""
    
        Calculate the total sum after incrementing the current count by 1.
    
        Args:
            count (int): Current count
            sum_so_far (int): Current total sum
    
        Returns:
            str: The new total sum after calculation.
    
        Example Output:
            "6"
        """
        new_sum = sum_so_far + (count + 1)
        return str(new_sum)

    def CheckCondition(self, sum_so_far: int, count: int):
        r"""
    
        Check if sum_so_far + (count + 1) is less than or equal to N.
    
        Args:
            sum_so_far (int): Current total sum
            count (int): Current count
    
        Returns:
            str: "True" if the condition is met, "False" otherwise.
    
        Example Output:
            "True"
        """
        condition = sum_so_far + (count + 1) <= self.N
        return str(condition)

    def IncrementCount(self, count: int):
        r"""
    
        Increment the count by 1.
    
        Args:
            count (int): Current count
    
        Returns:
            str: The new count after increment.
    
        Example Output:
            "3"
        """
        new_count = count + 1
        return str(new_count)

    def Observe(self):
        r"""
    
        Obtain the number of teams N in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current number of teams N.
    
        Example Output:
            "10"
        """
        return str(self.N)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
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
        N = int(self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1])
        count = 0
        sum_so_far = 0
        while True:
            condition = self.step(json.dumps({
                'name': 'CheckCondition',
                'parameters': {'sum_so_far': sum_so_far, 'count': count}
            }))[1]
            if condition == "False":
                break
            sum_so_far = int(self.step(json.dumps({
                'name': 'CalculateSum',
                'parameters': {'count': count, 'sum_so_far': sum_so_far}
            }))[1])
            count = int(self.step(json.dumps({
                'name': 'IncrementCount',
                'parameters': {'count': count}
            }))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = TeamAssignmentEnv.from_env_str("TeamAssignmentEnv@{\"N\": 10}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = TeamAssignmentEnv.from_env_str("TeamAssignmentEnv@{\"N\": 15}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    env = TeamAssignmentEnv.from_env_str("TeamAssignmentEnv@{\"N\": 20}")
    print(env.solve())
    print("step count:", env.step_count)