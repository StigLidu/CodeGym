# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class StringReductionEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.PROCESS_CHARACTER = 0
        self.GET_STACK_LENGTH = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "GetStackLength": self.GET_STACK_LENGTH,
            "Observe": self.OBSERVE,
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
        prefix = "StringReductionEnv@"
        if not env_str.startswith(prefix):
            return None
        return StringReductionEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.original_string = options.get("string", "")
        self.stack = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        stack = []
        for char in self.original_string:
            if stack and stack[-1] == char:
                stack.append(char)
            elif stack and stack[-1] != char:
                stack.pop()
            else:
                stack.append(char)
        return len(stack)

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
                if "char" in params:
                    char = params["char"]
                    msg = self.ProcessCharacter(char)
                else:
                    msg = "Error: 'char' parameter is missing for PROCESS_CHARACTER action."
            
            elif action_code == self.GET_STACK_LENGTH:
                msg = self.GetStackLength()
                
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
    def ProcessCharacter(self, char: str):
        r"""
    
        Process a character, adding it to the stack or removing it together with the top character of the stack according to the rules.
    
        Args:
            char (str): The single character to be processed.
    
        Returns:
            str: A description of the stack state after processing.
    
        Example Output:
            "Stack after processing 'a': ['a']"
        """
        if self.stack and self.stack[-1] == char:
            self.stack.append(char)
        elif self.stack and self.stack[-1] != char:
            self.stack.pop()
        else:
            self.stack.append(char)
        return f"Stack after processing '{char}': {self.stack}"

    def GetStackLength(self):
        r"""
    
        Get the current length of the stack.
    
        Args:
            None
    
        Returns:
            str: The current length of the stack.
    
        Example Output:
            "2"
        """
        return str(len(self.stack))

    def Observe(self):
        r"""
    
        Return the observation information of the current environment, including the original string and the current stack state.
    
        Args:
            None
    
        Returns:
            str: Information describing the current state of the environment.
    
        Example Output:
            "Original string: 'abab', Current stack: []"
        """
        return f"Original string: '{self.original_string}', Current stack: {self.stack}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, which is the length of the remaining string.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 0, Reference answer: 0, Result: Correct, reward=1"
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
        observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        original_str = observe_info.split("Original string: '")[1].split("', Current stack:")[0]
        
        for char in original_str:
            self.step(json.dumps({'name': 'ProcessCharacter', 'parameters': {'char': char}}))
        
        stack_length = int(self.step(json.dumps({'name': 'GetStackLength', 'parameters': {}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': stack_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "abab"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "abcabca"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_string = "aa"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)