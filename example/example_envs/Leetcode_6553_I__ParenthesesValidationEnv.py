# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class ParenthesesValidationEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.PROCESS_CHARACTER = 1
        self.CALCULATE_REMAINING = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "CalculateRemainingOperations": self.CALCULATE_REMAINING,
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
        prefix = "ParenthesesValidationEnv@"
        if not env_str.startswith(prefix):
            return None
        return ParenthesesValidationEnv(env_str=env_str)

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
        balance = 0
        operations = 0
        
        for char in self.s:
            if char == '(':
                balance += 1
            else:
                if balance > 0:
                    balance -= 1
                else:
                    operations += 1
        
        operations += balance
        return operations

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
            
            elif action_code == self.PROCESS_CHARACTER:
                if "char" in params and "current_balance" in params:
                    char = params["char"]
                    current_balance = params["current_balance"]
                    msg = self.ProcessCharacter(char, current_balance)
                else:
                    msg = "Error: 'char' or 'current_balance' parameter is missing for PROCESS_CHARACTER action."
            
            elif action_code == self.CALCULATE_REMAINING:
                if "remaining_balance" in params:
                    remaining_balance = params["remaining_balance"]
                    msg = self.CalculateRemainingOperations(remaining_balance)
                else:
                    msg = "Error: 'remaining_balance' parameter is missing for CALCULATE_REMAINING action."
            
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
    
        Obtain the current bracket sequence.
    
        Args:
            None
    
        Returns:
            str: The current bracket sequence.
    
        Example Output:
            "(()))("
        """
        return self.s

    def ProcessCharacter(self, char: str, current_balance: int):
        r"""
    
        Process a single bracket character, update the balance value, and determine if an operation is needed.
    
        Args:
            char (str): The bracket character to be processed, either '(' or ')'.
            current_balance (int): The current bracket balance value.
    
        Returns:
            str: A JSON string containing the new balance value and whether an operation is needed.
    
        Example Output:
            "{\"new_balance\": 1, \"operation_needed\": false}"
        """
        new_balance = current_balance
        operation_needed = False
        
        if char == '(':
            new_balance += 1
        else:
            if new_balance > 0:
                new_balance -= 1
            else:
                operation_needed = True
        
        result = {
            "new_balance": new_balance,
            "operation_needed": operation_needed
        }
        return json.dumps(result)

    def CalculateRemainingOperations(self, remaining_balance: int):
        r"""
    
        Calculate the number of operations corresponding to the remaining balance value.
    
        Args:
            remaining_balance (int): The final balance value after processing all characters.
    
        Returns:
            str: The number of operations corresponding to the remaining balance value.
    
        Example Output:
            "2"
        """
        return str(remaining_balance)

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return result information.
    
        Args:
            answer (int): The minimum number of operations submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        import json
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        current_balance = 0
        operation_count = 0
        for char in s:
            res = self.step(json.dumps({
                'name': 'ProcessCharacter',
                'parameters': {'char': char, 'current_balance': current_balance}
            }))[1]
            res_dict = json.loads(res)
            new_balance = res_dict['new_balance']
            operation_needed = res_dict['operation_needed']
            if operation_needed:
                operation_count += 1
            current_balance = new_balance
        remaining_ops = int(self.step(json.dumps({
            'name': 'CalculateRemainingOperations',
            'parameters': {'remaining_balance': current_balance}
        }))[1])
        total_ops = operation_count + remaining_ops
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total_ops}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_case1 = "())"
    env = ParenthesesValidationEnv.from_env_str(f"ParenthesesValidationEnv@{{\"s\": \"{test_case1}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_case2 = "(()))("
    env = ParenthesesValidationEnv.from_env_str(f"ParenthesesValidationEnv@{{\"s\": \"{test_case2}\"}}")
    print(env.solve())
    print("step count:", env.step_count)