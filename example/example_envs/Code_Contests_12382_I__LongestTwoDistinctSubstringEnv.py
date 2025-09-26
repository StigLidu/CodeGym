# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class LongestTwoDistinctSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.DONE = 1
        self.GET_STRING_LENGTH = 2
        self.GET_CHAR_AT_POSITION = 3
        self.UPDATE_CHAR_MAP = 4
        self.CHECK_CHAR_MAP_SIZE = 5
        self.DECREMENT_CHAR_COUNT = 6
        self.DELETE_CHAR_FROM_MAP = 7
        self.CALCULATE_WINDOW_LENGTH = 8
        self.UPDATE_MAX_LENGTH = 9

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "Done": self.DONE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "GetCharAtPosition": self.GET_CHAR_AT_POSITION,
            "UpdateCharMap": self.UPDATE_CHAR_MAP,
            "CheckCharMapSize": self.CHECK_CHAR_MAP_SIZE,
            "DecrementCharCount": self.DECREMENT_CHAR_COUNT,
            "DeleteCharFromMap": self.DELETE_CHAR_FROM_MAP,
            "CalculateWindowLength": self.CALCULATE_WINDOW_LENGTH,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH
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
        prefix = "LongestTwoDistinctSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestTwoDistinctSubstringEnv(env_str=env_str)

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
        if not self.s:
            return 0

        left = 0
        max_length = 0
        char_map = {}

        for right in range(len(self.s)):
            char = self.s[right]
            if char in char_map:
                char_map[char] += 1
            else:
                char_map[char] = 1

            while len(char_map) > 2:
                left_char = self.s[left]
                char_map[left_char] -= 1
                if char_map[left_char] == 0:
                    del char_map[left_char]
                left += 1

            max_length = max(max_length, right - left + 1)
        
        return max_length

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
                
            elif action_code == self.DONE:
                if "answer" in params:
                    answer = params["answer"]
                    msg = self.Done(answer)
                else:
                    msg = "Error: 'answer' parameter is missing for DONE action."
                    
            elif action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
                
            elif action_code == self.GET_CHAR_AT_POSITION:
                if "position" in params:
                    position = params["position"]
                    msg = self.GetCharAtPosition(position)
                else:
                    msg = "Error: 'position' parameter is missing for GET_CHAR_AT_POSITION action."
                    
            elif action_code == self.UPDATE_CHAR_MAP:
                if "char" in params and "char_map" in params:
                    char = params["char"]
                    char_map = params["char_map"]
                    msg = self.UpdateCharMap(char, char_map)
                else:
                    msg = "Error: 'char' or 'char_map' parameter is missing for UPDATE_CHAR_MAP action."
                    
            elif action_code == self.CHECK_CHAR_MAP_SIZE:
                if "char_map" in params:
                    char_map = params["char_map"]
                    msg = self.CheckCharMapSize(char_map)
                else:
                    msg = "Error: 'char_map' parameter is missing for CHECK_CHAR_MAP_SIZE action."
                    
            elif action_code == self.DECREMENT_CHAR_COUNT:
                if "char" in params and "char_map" in params:
                    char = params["char"]
                    char_map = params["char_map"]
                    msg = self.DecrementCharCount(char, char_map)
                else:
                    msg = "Error: 'char' or 'char_map' parameter is missing for DECREMENT_CHAR_COUNT action."
                    
            elif action_code == self.DELETE_CHAR_FROM_MAP:
                if "char" in params and "char_map" in params:
                    char = params["char"]
                    char_map = params["char_map"]
                    msg = self.DeleteCharFromMap(char, char_map)
                else:
                    msg = "Error: 'char' or 'char_map' parameter is missing for DELETE_CHAR_FROM_MAP action."
                    
            elif action_code == self.CALCULATE_WINDOW_LENGTH:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateWindowLength(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_WINDOW_LENGTH action."
                    
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "current_max" in params and "window_length" in params:
                    current_max = params["current_max"]
                    window_length = params["window_length"]
                    msg = self.UpdateMaxLength(current_max, window_length)
                else:
                    msg = "Error: 'current_max' or 'window_length' parameter is missing for UPDATE_MAX_LENGTH action."
                    
        except Exception as e:
            msg = f"Error: {str(e)}"

        return True, msg

    # All the actions of the environment
    def Observe(self):
        r"""
    
        Obtain the string information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "abcba"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 3, Reference answer: 3, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def GetStringLength(self):
        r"""
    
        Get the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the string.
    
        Example Output:
            "5"
        """
        return str(len(self.s))

    def GetCharAtPosition(self, position: int):
        r"""
    
        Get the character at the specified position in the string.
    
        Args:
            position (int): The position index of the character to be obtained.
    
        Returns:
            str: The character at the specified position.
    
        Example Output:
            "b"
        """
        if 0 <= position < len(self.s):
            return self.s[position]
        return ""

    def UpdateCharMap(self, char: str, char_map: dict):
        r"""
    
        Update the character mapping table to increase the count of the specified character.
    
        Args:
            char (str): The character to be updated.
            char_map (dict): The current character mapping table.
    
        Returns:
            str: The updated character mapping table.
    
        Example Output:
            "{\"a\": 1, \"b\": 2}"
        """
        if char in char_map:
            char_map[char] += 1
        else:
            char_map[char] = 1
        return json.dumps(char_map)

    def CheckCharMapSize(self, char_map: dict):
        r"""
    
        Check the number of distinct characters in the character mapping table.
    
        Args:
            char_map (dict): The current character mapping table.
    
        Returns:
            str: The number of distinct characters in the character mapping table.
    
        Example Output:
            "2"
        """
        return str(len(char_map))

    def DecrementCharCount(self, char: str, char_map: dict):
        r"""
    
        Decrease the count of the specified character in the character mapping table.
    
        Args:
            char (str): The character whose count is to be decreased.
            char_map (dict): The current character mapping table.
    
        Returns:
            str: The updated character mapping table.
    
        Example Output:
            "{\"a\": 0, \"b\": 2}"
        """
        if char in char_map:
            char_map[char] -= 1
        return json.dumps(char_map)

    def DeleteCharFromMap(self, char: str, char_map: dict):
        r"""
    
        Delete the character with a count of 0 from the character mapping table.
    
        Args:
            char (str): The character to be deleted.
            char_map (dict): The current character mapping table.
    
        Returns:
            str: The updated character mapping table.
    
        Example Output:
            "{\"b\": 2}"
        """
        if char in char_map and char_map[char] == 0:
            del char_map[char]
        return json.dumps(char_map)

    def CalculateWindowLength(self, left: int, right: int):
        r"""
    
        Calculate the length of the current window.
    
        Args:
            left (int): The left boundary index of the window.
            right (int): The right boundary index of the window.
    
        Returns:
            str: The length of the window.
    
        Example Output:
            "3"
        """
        return str(right - left + 1)

    def UpdateMaxLength(self, current_max: int, window_length: int):
        r"""
    
        Update the maximum window length.
    
        Args:
            current_max (int): The current maximum window length.
            window_length (int): The newly calculated window length.
    
        Returns:
            str: The updated maximum window length.
    
        Example Output:
            "3"
        """
        return str(max(current_max, window_length))

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        str_len_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        str_len = int(str_len_str)
        
        if str_len == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        left = 0
        max_length = 0
        char_map = {}
        
        for right in range(str_len):
            char_right_str = self.step(json.dumps({
                'name': 'GetCharAtPosition',
                'parameters': {'position': right}
            }))[1]
            char_right = char_right_str
            
            update_map_str = self.step(json.dumps({
                'name': 'UpdateCharMap',
                'parameters': {'char': char_right, 'char_map': char_map}
            }))[1]
            char_map = json.loads(update_map_str.replace("'", "\""))
            
            while True:
                map_size_str = self.step(json.dumps({
                    'name': 'CheckCharMapSize',
                    'parameters': {'char_map': char_map}
                }))[1]
                map_size = int(map_size_str)
                
                if map_size <= 2:
                    break
                
                char_left_str = self.step(json.dumps({
                    'name': 'GetCharAtPosition',
                    'parameters': {'position': left}
                }))[1]
                char_left = char_left_str
                
                decremented_map_str = self.step(json.dumps({
                    'name': 'DecrementCharCount',
                    'parameters': {'char': char_left, 'char_map': char_map}
                }))[1]
                char_map = json.loads(decremented_map_str.replace("'", "\""))
                
                if char_map[char_left] == 0:
                    deleted_map_str = self.step(json.dumps({
                        'name': 'DeleteCharFromMap',
                        'parameters': {'char': char_left, 'char_map': char_map}
                    }))[1]
                    char_map = json.loads(deleted_map_str.replace("'", "\""))
                
                left += 1
            
            window_len_str = self.step(json.dumps({
                'name': 'CalculateWindowLength',
                'parameters': {'left': left, 'right': right}
            }))[1]
            window_len = int(window_len_str)
            
            max_len_str = self.step(json.dumps({
                'name': 'UpdateMaxLength',
                'parameters': {'current_max': max_length, 'window_length': window_len}
            }))[1]
            max_length = int(max_len_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - sample input
    print("Test Case 1:")
    env1 = LongestTwoDistinctSubstringEnv.from_env_str('LongestTwoDistinctSubstringEnv@{"s": "abcba"}')
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2 - random string
    print("\nTest Case 2:")
    test_string = "abaccc"  # The longest substring is "accc" with length 4
    env2 = LongestTwoDistinctSubstringEnv.from_env_str(f'LongestTwoDistinctSubstringEnv@{{"s": "{test_string}"}}')
    print(env2.solve())
    print("step count:", env2.step_count)