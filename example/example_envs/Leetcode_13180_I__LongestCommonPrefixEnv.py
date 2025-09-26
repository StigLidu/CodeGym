# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast

class LongestCommonPrefixEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_FIRST_STRING = 1
        self.COMPARE_PREFIX_WITH_STRING = 2
        self.SHORTEN_PREFIX = 3
        self.IS_PREFIX_EMPTY = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetFirstString": self.GET_FIRST_STRING,
            "ComparePrefixWithString": self.COMPARE_PREFIX_WITH_STRING,
            "ShortenPrefix": self.SHORTEN_PREFIX,
            "IsPrefixEmpty": self.IS_PREFIX_EMPTY,
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
        prefix = "LongestCommonPrefixEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestCommonPrefixEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.strs = options.get("strs", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.strs:
            return ""
        
        prefix = self.strs[0]
        
        for string in self.strs[1:]:
            while string[:len(prefix)] != prefix:
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

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
            
            elif action_code == self.GET_FIRST_STRING:
                msg = self.GetFirstString()
                
            elif action_code == self.COMPARE_PREFIX_WITH_STRING:
                if "prefix" in params and "string" in params:
                    prefix = params["prefix"]
                    string = params["string"]
                    msg = self.ComparePrefixWithString(prefix, string)
                else:
                    msg = "Error: 'prefix' or 'string' parameter is missing for COMPARE_PREFIX_WITH_STRING action."
                    
            elif action_code == self.SHORTEN_PREFIX:
                if "prefix" in params:
                    prefix = params["prefix"]
                    msg = self.ShortenPrefix(prefix)
                else:
                    msg = "Error: 'prefix' parameter is missing for SHORTEN_PREFIX action."
                    
            elif action_code == self.IS_PREFIX_EMPTY:
                if "prefix" in params:
                    prefix = params["prefix"]
                    msg = self.IsPrefixEmpty(prefix)
                else:
                    msg = "Error: 'prefix' parameter is missing for IS_PREFIX_EMPTY action."
                    
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
    
        Obtain the list of strings in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of strings in the environment.
    
        Example Output:
            "[\"flower\", \"flow\", \"flight\"]"
        """
        return json.dumps(self.strs)

    def GetFirstString(self):
        r"""
    
        Obtain the first string in the list of strings.
    
        Args:
            None
    
        Returns:
            str: The first string in the list; returns an empty string if the list is empty.
    
        Example Output:
            "flower"
        """
        if self.strs:
            return self.strs[0]
        return ""

    def ComparePrefixWithString(self, prefix: str, string: str):
        r"""
    
        Compare the prefix with the specified string to check if the prefix is a prefix of the string.
    
        Args:
            prefix (str): The prefix to check.
            string (str): The string to compare.
    
        Returns:
            str: "True" if the prefix is a prefix of the string, otherwise "False".
    
        Example Output:
            "True"
        """
        return str(string[:len(prefix)] == prefix)

    def ShortenPrefix(self, prefix: str):
        r"""
    
        Shorten the prefix by removing the last character.
    
        Args:
            prefix (str): The prefix to shorten.
    
        Returns:
            str: The shortened prefix.
    
        Example Output:
            "flo"
        """
        return prefix[:-1] if prefix else ""

    def IsPrefixEmpty(self, prefix: str):
        r"""
    
        Check if the prefix is empty.
    
        Args:
            prefix (str): The prefix to check.
    
        Returns:
            str: "True" if the prefix is empty, otherwise "False".
    
        Example Output:
            "False"
        """
        return str(len(prefix) == 0)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: fl, Reference answer: fl, Result: Correct, reward=1"
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
        prefix = self.step(json.dumps({'name': 'GetFirstString', 'parameters': {}}))[1]
        
        str_list = ast.literal_eval(self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1])
        
        for s in str_list[1:]:
            is_prefix = self.step(json.dumps({
                'name': 'ComparePrefixWithString',
                'parameters': {'prefix': prefix, 'string': s}
            }))[1]
            
            while is_prefix == "False":
                prefix = self.step(json.dumps({
                    'name': 'ShortenPrefix',
                    'parameters': {'prefix': prefix}
                }))[1]
                
                if self.step(json.dumps({
                    'name': 'IsPrefixEmpty',
                    'parameters': {'prefix': prefix}
                }))[1] == "True":
                    break
                
                is_prefix = self.step(json.dumps({
                    'name': 'ComparePrefixWithString',
                    'parameters': {'prefix': prefix, 'string': s}
                }))[1]
            
            if self.step(json.dumps({
                'name': 'IsPrefixEmpty',
                'parameters': {'prefix': prefix}
            }))[1] == "True":
                break
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': prefix}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_strs1 = ["flower", "flow", "flight"]
    env1 = LongestCommonPrefixEnv.from_env_str(f"LongestCommonPrefixEnv@{{\"strs\": {test_strs1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_strs2 = ["dog", "racecar", "car"]
    env2 = LongestCommonPrefixEnv.from_env_str(f"LongestCommonPrefixEnv@{{\"strs\": {test_strs2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)