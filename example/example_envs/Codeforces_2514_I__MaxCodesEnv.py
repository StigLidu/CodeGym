# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MaxCodesEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_INTERVAL = 0
        self.ADD_INTERVAL = 1
        self.GET_SELECTED_COUNT = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateInterval": self.CALCULATE_INTERVAL,
            "AddInterval": self.ADD_INTERVAL,
            "GetSelectedCount": self.GET_SELECTED_COUNT,
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
        prefix = "MaxCodesEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxCodesEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.codes = options.get("codes", [])
        self.selected_intervals = set()
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        intervals = set()
        for _, timestamp in self.codes:
            interval = timestamp // 5
            if interval not in intervals:
                intervals.add(interval)
        return len(intervals)

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
            
            if action_code == self.CALCULATE_INTERVAL:
                if "timestamp" in params:
                    timestamp = params["timestamp"]
                    msg = self.CalculateInterval(timestamp)
                else:
                    msg = "Error: 'timestamp' parameter is missing for CALCULATE_INTERVAL action."
            
            elif action_code == self.ADD_INTERVAL:
                if "interval" in params:
                    interval = params["interval"]
                    msg = self.AddInterval(interval)
                else:
                    msg = "Error: 'interval' parameter is missing for ADD_INTERVAL action."
                    
            elif action_code == self.GET_SELECTED_COUNT:
                msg = self.GetSelectedCount()
                
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
    def CalculateInterval(self, timestamp: int):
        r"""
    
        Calculate the five-minute interval that the given timestamp belongs to.
    
        Args:
            timestamp (int): The timestamp of the code.
    
        Returns:
            str: The five-minute interval number that the timestamp belongs to.
    
        Example Output:
            "1"
        """
        interval = timestamp // 5
        return str(interval)

    def AddInterval(self, interval: int):
        r"""
    
        Add an interval to the selected set; the addition is successful if the interval does not exist.
    
        Args:
            interval (int): The five-minute interval number to be added.
    
        Returns:
            str: The result of the addition operation, "Added" indicates successful addition, and "Already exists" indicates it already exists.
    
        Example Output:
            "Added"
        """
        if interval not in self.selected_intervals:
            self.selected_intervals.add(interval)
            return "Added"
        return "Already exists"

    def GetSelectedCount(self):
        r"""
    
        Get the number of selected intervals.
    
        Args:
            None
    
        Returns:
            str: The number of selected intervals.
    
        Example Output:
            "3"
        """
        return str(len(self.selected_intervals))

    def Observe(self):
        r"""
    
        Get all code timestamps in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of all code timestamps.
    
        Example Output:
            "[3, 7, 8, 12]"
        """
        timestamps = [timestamp for _, timestamp in self.codes]
        return json.dumps(timestamps)

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
        timestamps_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        timestamps = ast.literal_eval(timestamps_str)
        
        for ts in timestamps:
            interval_str = self.step(json.dumps({'name': 'CalculateInterval', 'parameters': {'timestamp': ts}}))[1]
            interval = int(interval_str)
            self.step(json.dumps({'name': 'AddInterval', 'parameters': {'interval': interval}}))
        
        count_str = self.step(json.dumps({'name': 'GetSelectedCount', 'parameters': {}}))[1]
        count = int(count_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (from example)
    print("Test Case 1:")
    test_codes1 = [("code1", 3), ("code2", 7), ("code3", 8), ("code4", 12)]
    env1 = MaxCodesEnv.from_env_str(f"MaxCodesEnv@{{\"codes\": {test_codes1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_codes2 = [("a", 0), ("b", 4), ("c", 5), ("d", 9), ("e", 10), ("f", 14)]
    env2 = MaxCodesEnv.from_env_str(f"MaxCodesEnv@{{\"codes\": {test_codes2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)