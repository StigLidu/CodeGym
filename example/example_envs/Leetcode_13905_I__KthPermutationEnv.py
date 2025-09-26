# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import math
import ast
import json

class KthPermutationEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_FACTORIAL = 0
        self.GET_DIVMOD = 1
        self.POP_NUMBER_BY_INDEX = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateFactorial": self.CALCULATE_FACTORIAL,
            "GetDivmod": self.GET_DIVMOD,
            "PopNumberByIndex": self.POP_NUMBER_BY_INDEX,
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
        prefix = "KthPermutationEnv@"
        if not env_str.startswith(prefix):
            return None
        return KthPermutationEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 3)
        self.k = options.get("k", 1)
        self.available_numbers = list(range(1, self.n + 1))
        self.current_permutation = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        numbers = list(range(1, self.n + 1))
        permutation = []
        k = self.k - 1  # convert k to zero-indexed
        
        n = self.n
        while n > 0:
            n -= 1
            index, k = divmod(k, math.factorial(n))
            permutation.append(str(numbers.pop(index)))
        
        return ''.join(permutation)

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
            
            if action_code == self.CALCULATE_FACTORIAL:
                if "number" in params:
                    number = params["number"]
                    msg = self.CalculateFactorial(number)
                else:
                    msg = "Error: 'number' parameter is missing for CALCULATE_FACTORIAL action."
            
            elif action_code == self.GET_DIVMOD:
                if "dividend" in params and "divisor" in params:
                    dividend = params["dividend"]
                    divisor = params["divisor"]
                    msg = self.GetDivmod(dividend, divisor)
                else:
                    msg = "Error: 'dividend' or 'divisor' parameter is missing for GET_DIVMOD action."
            
            elif action_code == self.POP_NUMBER_BY_INDEX:
                if "index" in params:
                    index = params["index"]
                    msg = self.PopNumberByIndex(index)
                else:
                    msg = "Error: 'index' parameter is missing for POP_NUMBER_BY_INDEX action."
            
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
    def CalculateFactorial(self, number: int):
        r"""
    
        Calculate the factorial of a given number.
    
        Args:
            number (int): The number for which to calculate the factorial.
    
        Returns:
            str: The factorial result of the number.
    
        Example Output:
            "6"
        """
        return str(math.factorial(number))

    def GetDivmod(self, dividend: int, divisor: int):
        r"""
    
        Calculate the quotient and remainder of the dividend divided by the divisor.
    
        Args:
            dividend (int): The dividend.
            divisor (int): The divisor.
    
        Returns:
            str: A tuple string containing the quotient and remainder.
    
        Example Output:
            "(1, 2)"
        """
        return str(divmod(dividend, divisor))

    def PopNumberByIndex(self, index: int):
        r"""
    
        Pop the number at the specified index from the list of available numbers and add it to the current permutation.
    
        Args:
            index (int): The index of the number to be popped.
    
        Returns:
            str: The popped number.
    
        Example Output:
            "2"
        """
        number = self.available_numbers.pop(index)
        self.current_permutation.append(str(number))
        return str(number)

    def Observe(self):
        r"""
    
        Return the status information of the current environment, including n, k, the list of available numbers, and the current permutation.
    
        Args:
            None
    
        Returns:
            str: Information describing the current environment status.
    
        Example Output:
            "n=3, k=3, available_numbers=[1,2,3], current_permutation=[]"
        """
        return f"n={self.n}, k={self.k}, available_numbers={self.available_numbers}, current_permutation={self.current_permutation}"

    def Done(self, answer: str):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (str): The permutation string submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 213, Reference answer: 213, Result: Correct, reward=1"
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
        import ast
    
        obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = int(obs.split('n=')[1].split(',')[0])
        k = int(obs.split('k=')[1].split(',')[0])
        k -= 1
    
        for i in range(n, 0, -1):
            if i == 1:
                self.step(json.dumps({'name': 'PopNumberByIndex', 'parameters': {'index': 0}}))
            else:
                factorial_str = self.step(json.dumps({'name': 'CalculateFactorial', 'parameters': {'number': i - 1}}))[1]
                factorial = int(factorial_str)
                divmod_str = self.step(json.dumps({'name': 'GetDivmod', 'parameters': {'dividend': k, 'divisor': factorial}}))[1]
                q, r = ast.literal_eval(divmod_str)
                self.step(json.dumps({'name': 'PopNumberByIndex', 'parameters': {'index': q}}))
                k = r
    
        final_obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        current_perm = final_obs.split('current_permutation=')[1]
        perm_list = ast.literal_eval(current_perm)
        answer = ''.join(map(str, perm_list))
    
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = KthPermutationEnv.from_env_str("KthPermutationEnv@{\"n\": 3, \"k\": 3}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = KthPermutationEnv.from_env_str("KthPermutationEnv@{\"n\": 4, \"k\": 9}")
    print(env.solve())
    print("step count:", env.step_count)