# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from itertools import product

class NextClosestTimeEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CONVERT_TO_MINUTES = 1
        self.INCREMENT_MINUTES = 2
        self.CONVERT_TO_TIME_STR = 3
        self.CHECK_TIME_VALIDITY = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "ConvertToMinutes": self.CONVERT_TO_MINUTES,
            "IncrementMinutes": self.INCREMENT_MINUTES,
            "ConvertToTimeStr": self.CONVERT_TO_TIME_STR,
            "CheckTimeValidity": self.CHECK_TIME_VALIDITY,
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
        prefix = "NextClosestTimeEnv@"
        if not env_str.startswith(prefix):
            return None
        return NextClosestTimeEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.current_time = options.get("time", "00:00")
        self.allowed_digits = {int(x) for x in self.current_time if x != ':'}
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        current_minutes = int(self.current_time[:2]) * 60 + int(self.current_time[3:])
        allowed_digits = self.allowed_digits
        
        while True:
            current_minutes = (current_minutes + 1) % (24 * 60)
            h, m = divmod(current_minutes, 60)
            next_time = f"{h:02}:{m:02}"
            if all(int(x) in allowed_digits for x in next_time if x != ':'):
                return next_time

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
            
            elif action_code == self.CONVERT_TO_MINUTES:
                if "time_str" in params:
                    time_str = params["time_str"]
                    msg = self.ConvertToMinutes(time_str)
                else:
                    msg = "Error: 'time_str' parameter is missing for CONVERT_TO_MINUTES action."
            
            elif action_code == self.INCREMENT_MINUTES:
                if "minutes" in params:
                    minutes = params["minutes"]
                    msg = self.IncrementMinutes(minutes)
                else:
                    msg = "Error: 'minutes' parameter is missing for INCREMENT_MINUTES action."
            
            elif action_code == self.CONVERT_TO_TIME_STR:
                if "minutes" in params:
                    minutes = params["minutes"]
                    msg = self.ConvertToTimeStr(minutes)
                else:
                    msg = "Error: 'minutes' parameter is missing for CONVERT_TO_TIME_STR action."
            
            elif action_code == self.CHECK_TIME_VALIDITY:
                if "time_str" in params:
                    time_str = params["time_str"]
                    msg = self.CheckTimeValidity(time_str)
                else:
                    msg = "Error: 'time_str' parameter is missing for CHECK_TIME_VALIDITY action."
            
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
    
        Get the time displayed on the current clock and the allowed digits.
    
        Args:
            None
    
        Returns:
            str: The current time and allowed digits, in JSON format string.
    
        Example Output:
            "{\"current_time\": \"19:34\", \"allowed_digits\": [1, 9, 3, 4]}"
        """
        observation = {
            "current_time": self.current_time,
            "allowed_digits": list(self.allowed_digits)
        }
        return json.dumps(observation)

    def ConvertToMinutes(self, time_str: str):
        r"""
    
        Convert a time string in HH:MM format to the number of minutes.
    
        Args:
            time_str (str): A time string in HH:MM format.
    
        Returns:
            str: The converted number of minutes.
    
        Example Output:
            "1174"
        """
        hours = int(time_str[:2])
        minutes = int(time_str[3:])
        total_minutes = hours * 60 + minutes
        return str(total_minutes)

    def IncrementMinutes(self, minutes: int):
        r"""
    
        Increment the number of minutes by 1, taking modulo if exceeding 24 hours.
    
        Args:
            minutes (int): The current number of minutes.
    
        Returns:
            str: The incremented number of minutes.
    
        Example Output:
            "1175"
        """
        new_minutes = (minutes + 1) % (24 * 60)
        return str(new_minutes)

    def ConvertToTimeStr(self, minutes: int):
        r"""
    
        Convert the number of minutes to a time string in HH:MM format.
    
        Args:
            minutes (int): The number of minutes to convert.
    
        Returns:
            str: A time string in HH:MM format.
    
        Example Output:
            "19:35"
        """
        h, m = divmod(minutes, 60)
        return f"{h:02}:{m:02}"

    def CheckTimeValidity(self, time_str: str):
        r"""
    
        Check if the time string can be formed using the allowed digits.
    
        Args:
            time_str (str): The time string in HH:MM format to check.
    
        Returns:
            str: "True" if valid, "False" if invalid.
    
        Example Output:
            "False"
        """
        digits_used = [int(x) for x in time_str if x != ':']
        valid = all(d in self.allowed_digits for d in digits_used)
        return str(valid)

    def Done(self, answer: str):
        r"""
    
        Verify if the final answer is correct and return the result information.
    
        Args:
            answer (str): The time string in HH:MM format submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 19:39, Reference answer: 19:39, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
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
        current_time = observe_data['current_time']
        
        current_minutes = int(self.step(json.dumps({'name': 'ConvertToMinutes', 'parameters': {'time_str': current_time}}))[1])
        
        while True:
            next_minutes = int(self.step(json.dumps({'name': 'IncrementMinutes', 'parameters': {'minutes': current_minutes}}))[1])
            next_time_str = self.step(json.dumps({'name': 'ConvertToTimeStr', 'parameters': {'minutes': next_minutes}}))[1]
            is_valid = self.step(json.dumps({'name': 'CheckTimeValidity', 'parameters': {'time_str': next_time_str}}))[1]
            if is_valid == "True":
                return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': next_time_str}}))[1]
            current_minutes = next_minutes
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = NextClosestTimeEnv.from_env_str("NextClosestTimeEnv@{\"time\": \"19:34\"}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = NextClosestTimeEnv.from_env_str("NextClosestTimeEnv@{\"time\": \"23:59\"}")
    print(env.solve())
    print("step count:", env.step_count)