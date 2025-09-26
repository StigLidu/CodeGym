# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class ParenthesesBalanceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.PROCESS_CHARACTER = 0
        self.CALCULATE_RESULT = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "CalculateResult": self.CALCULATE_RESULT,
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
        prefix = "ParenthesesBalanceEnv@"
        if not env_str.startswith(prefix):
            return None
        return ParenthesesBalanceEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        open_needed, close_needed = 0, 0
        
        for char in self.s:
            if char == '(':
                close_needed += 1
            else:  # char == ')'
                if close_needed > 0:
                    close_needed -= 1
                else:
                    open_needed += 1
        
        return open_needed + close_needed

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
            
            if action_code == self.PROCESS_CHARACTER:
                if "char" in params and "current_open" in params and "current_close" in params:
                    char = params["char"]
                    current_open = params["current_open"]
                    current_close = params["current_close"]
                    msg = self.ProcessCharacter(char, current_open, current_close)
                else:
                    msg = "Error: 'char', 'current_open' or 'current_close' parameter is missing for PROCESS_CHARACTER action."
            
            elif action_code == self.CALCULATE_RESULT:
                if "open_needed" in params and "close_needed" in params:
                    open_needed = params["open_needed"]
                    close_needed = params["close_needed"]
                    msg = self.CalculateResult(open_needed, close_needed)
                else:
                    msg = "Error: 'open_needed' or 'close_needed' parameter is missing for CALCULATE_RESULT action."
                    
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
    def ProcessCharacter(self, char: str, current_open: int, current_close: int):
        r"""
    
        Processes a single parenthesis character and updates the required number of opening and closing parentheses.
    
        Args:
            char (str): The parenthesis character to be processed, either '(' or ')'.
            current_open (int): The current number of required opening parentheses.
            current_close (int): The current number of required closing parentheses.
    
        Returns:
            str: The processed number of required opening and closing parentheses, formatted as "open_needed,close_needed".
    
        Example Output:
            "0,1"
        """
        open_needed, close_needed = current_open, current_close
        
        if char == '(':
            close_needed += 1
        else:  # char == ')'
            if close_needed > 0:
                close_needed -= 1
            else:
                open_needed += 1
                
        return f"{open_needed},{close_needed}"

    def CalculateResult(self, open_needed: int, close_needed: int):
        r"""
    
        Calculates the minimum number of operations required to balance the parenthesis string.
    
        Args:
            open_needed (int): The number of required opening parentheses.
            close_needed (int): The number of required closing parentheses.
    
        Returns:
            str: The minimum number of operations.
    
        Example Output:
            "4"
        """
        return str(open_needed + close_needed)

    def Observe(self):
        r"""
    
        Retrieves the parenthesis string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current parenthesis string.
    
        Example Output:
            "))(("
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns result information.
    
        Args:
            answer (int): The minimum number of operations submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        open_needed = 0
        close_needed = 0
        for char in s:
            result = self.step(json.dumps({
                'name': 'ProcessCharacter',
                'parameters': {
                    'char': char,
                    'current_open': open_needed,
                    'current_close': close_needed
                }
            }))[1]
            open_needed, close_needed = map(int, result.split(','))
        min_operations = int(self.step(json.dumps({
            'name': 'CalculateResult',
            'parameters': {
                'open_needed': open_needed,
                'close_needed': close_needed
            }
        }))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': min_operations}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "))(("
    env = ParenthesesBalanceEnv.from_env_str(f"ParenthesesBalanceEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "()()("
    env = ParenthesesBalanceEnv.from_env_str(f"ParenthesesBalanceEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)