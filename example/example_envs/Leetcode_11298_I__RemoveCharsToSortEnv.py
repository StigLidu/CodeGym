# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class RemoveCharsToSortEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_K_GREATER_EQUAL_LENGTH = 1
        self.INITIALIZE_STACK = 2
        self.PROCESS_CHARACTER = 3
        self.REMOVE_REMAINING_K = 4
        self.FORM_RESULT_STRING = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckKGreaterEqualLength": self.CHECK_K_GREATER_EQUAL_LENGTH,
            "InitializeStack": self.INITIALIZE_STACK,
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "RemoveRemainingK": self.REMOVE_REMAINING_K,
            "FormResultString": self.FORM_RESULT_STRING,
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
        prefix = "RemoveCharsToSortEnv@"
        if not env_str.startswith(prefix):
            return None
        return RemoveCharsToSortEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.k = options.get("k", 0)
        self.stack = []
        self.current_index = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        n = len(self.s)
        if self.k >= n:
            return ""

        stack = []
        remaining_k = self.k
        for char in self.s:
            while remaining_k > 0 and stack and stack[-1] > char:
                stack.pop()
                remaining_k -= 1
            stack.append(char)
            
        while remaining_k > 0:
            stack.pop()
            remaining_k -= 1

        return ''.join(stack)

    # [Required] Define the step method of the environment
    def step(self, action: str):
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
            
            elif action_code == self.CHECK_K_GREATER_EQUAL_LENGTH:
                msg = self.CheckKGreaterEqualLength()
                
            elif action_code == self.INITIALIZE_STACK:
                msg = self.InitializeStack()
                
            elif action_code == self.PROCESS_CHARACTER:
                if "char" in params and "remaining_k" in params:
                    char = params["char"]
                    remaining_k = params["remaining_k"]
                    msg = self.ProcessCharacter(char, remaining_k)
                else:
                    msg = "Error: 'char' or 'remaining_k' parameter is missing for PROCESS_CHARACTER action."
                    
            elif action_code == self.REMOVE_REMAINING_K:
                if "remaining_k" in params:
                    remaining_k = params["remaining_k"]
                    msg = self.RemoveRemainingK(remaining_k)
                else:
                    msg = "Error: 'remaining_k' parameter is missing for REMOVE_REMAINING_K action."
                    
            elif action_code == self.FORM_RESULT_STRING:
                msg = self.FormResultString()
                
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
    
        Obtain the string in the current environment and the number of characters to be deleted.
    
        Args:
            None
    
        Returns:
            str: Information containing the current string and the k value.
    
        Example Output:
            "{\"s\": \"cbacdcbc\", \"k\": 4}"
        """
        return json.dumps({"s": self.s, "k": self.k})

    def CheckKGreaterEqualLength(self):
        r"""
    
        Check if k is greater than or equal to the length of the string.
    
        Args:
            None
    
        Returns:
            str: "True" indicates that k is greater than or equal to the string length, "False" otherwise.
    
        Example Output:
            "False"
        """
        return str(self.k >= len(self.s)).lower()

    def InitializeStack(self):
        r"""
    
        Initialize the stack used for processing the string.
    
        Args:
            None
    
        Returns:
            str: The state of the stack after initialization.
    
        Example Output:
            "[]"
        """
        self.stack = []
        return json.dumps(self.stack)

    def ProcessCharacter(self, char: str, remaining_k: int):
        r"""
    
        Process a single character and decide whether to pop the top element of the stack according to the rules.
    
        Args:
            char (str): The current character to be processed.
            remaining_k (int): The remaining number of characters that can be deleted.
    
        Returns:
            str: The state of the stack after processing and the remaining number of deletable characters.
    
        Example Output:
            "{\"stack\": [\"a\", \"b\"], \"remaining_k\": 2}"
        """
        remaining_k = int(remaining_k)
        
        while remaining_k > 0 and self.stack and self.stack[-1] > char:
            self.stack.pop()
            remaining_k -= 1
        self.stack.append(char)
        
        return json.dumps({"stack": self.stack, "remaining_k": remaining_k})

    def RemoveRemainingK(self, remaining_k: int):
        r"""
    
        If there are remaining deletable characters, delete from the end of the stack.
    
        Args:
            remaining_k (int): The remaining number of characters that can be deleted.
    
        Returns:
            str: The state of the stack after processing.
    
        Example Output:
            "[\"a\", \"c\", \"b\", \"c\"]"
        """
        remaining_k = int(remaining_k)
        
        while remaining_k > 0 and self.stack:
            self.stack.pop()
            remaining_k -= 1
            
        return json.dumps(self.stack)

    def FormResultString(self):
        r"""
    
        Combine the characters in the stack into the result string.
    
        Args:
            None
    
        Returns:
            str: The string composed of the characters in the stack.
    
        Example Output:
            "acbc"
        """
        return ''.join(self.stack)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: acbc, Reference answer: acbc, Result: Correct, reward=1"
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
        s = observe_data['s']
        k = observe_data['k']
        
        check_result = self.step(json.dumps({'name': 'CheckKGreaterEqualLength', 'parameters': {}}))[1]
        if check_result == "True":
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': ""}}))[1]
        
        self.step(json.dumps({'name': 'InitializeStack', 'parameters': {}}))
        
        remaining_k = k
        for char in s:
            process_result = self.step(json.dumps({
                'name': 'ProcessCharacter',
                'parameters': {'char': char, 'remaining_k': remaining_k}
            }))[1]
            process_data = json.loads(process_result)
            remaining_k = process_data['remaining_k']
        
        if remaining_k > 0:
            self.step(json.dumps({
                'name': 'RemoveRemainingK',
                'parameters': {'remaining_k': remaining_k}
            }))
        
        result_str = self.step(json.dumps({'name': 'FormResultString', 'parameters': {}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': result_str}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = RemoveCharsToSortEnv.from_env_str('RemoveCharsToSortEnv@{"s": "cbacdcbc", "k": 4}')
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = RemoveCharsToSortEnv.from_env_str('RemoveCharsToSortEnv@{"s": "abcd", "k": 2}')
    print(env.solve())
    print("step count:", env.step_count)