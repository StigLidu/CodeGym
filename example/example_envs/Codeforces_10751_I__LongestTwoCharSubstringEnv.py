import gymnasium
import ast
import json

class LongestTwoCharSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_STRING_LENGTH = 1
        self.UPDATE_CHAR_INDEX = 2
        self.CHECK_CHAR_COUNT = 3
        self.ADJUST_LEFT_POINTER = 4
        self.CALCULATE_CURRENT_LENGTH = 5
        self.UPDATE_MAX_LENGTH = 6
        self.DONE = 7

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "UpdateCharIndex": self.UPDATE_CHAR_INDEX,
            "CheckCharCount": self.CHECK_CHAR_COUNT,
            "AdjustLeftPointer": self.ADJUST_LEFT_POINTER,
            "CalculateCurrentLength": self.CALCULATE_CURRENT_LENGTH,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
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
        prefix = "LongestTwoCharSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestTwoCharSubstringEnv(env_str=env_str)

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
        recent_left_char_index = {}
        max_length = 0

        for right in range(len(self.s)):
            recent_left_char_index[self.s[right]] = right

            if len(recent_left_char_index) > 2:
                left_most_index = min(recent_left_char_index.values())
                del recent_left_char_index[self.s[left_most_index]]
                left = left_most_index + 1
            
            current_length = right - left + 1
            max_length = max(max_length, current_length)
        
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
            
            elif action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
                
            elif action_code == self.UPDATE_CHAR_INDEX:
                if "char" in params and "index" in params and "char_index_dict" in params:
                    char = params["char"]
                    index = params["index"]
                    char_index_dict = params["char_index_dict"]
                    msg = self.UpdateCharIndex(char, index, char_index_dict)
                else:
                    msg = "Error: 'char', 'index' or 'char_index_dict' parameter is missing for UPDATE_CHAR_INDEX action."
                    
            elif action_code == self.CHECK_CHAR_COUNT:
                if "char_index_dict" in params:
                    char_index_dict = params["char_index_dict"]
                    msg = self.CheckCharCount(char_index_dict)
                else:
                    msg = "Error: 'char_index_dict' parameter is missing for CHECK_CHAR_COUNT action."
                    
            elif action_code == self.ADJUST_LEFT_POINTER:
                if "char_index_dict" in params and "left" in params:
                    char_index_dict = params["char_index_dict"]
                    left = params["left"]
                    result = self.AdjustLeftPointer(char_index_dict, left)
                    msg = json.dumps(result)
                else:
                    msg = "Error: 'char_index_dict' or 'left' parameter is missing for ADJUST_LEFT_POINTER action."
                    
            elif action_code == self.CALCULATE_CURRENT_LENGTH:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateCurrentLength(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_CURRENT_LENGTH action."
                    
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "current_length" in params and "max_length" in params:
                    current_length = params["current_length"]
                    max_length = params["max_length"]
                    msg = self.UpdateMaxLength(current_length, max_length)
                else:
                    msg = "Error: 'current_length' or 'max_length' parameter is missing for UPDATE_MAX_LENGTH action."
                    
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
    
        Obtain the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "eceba"
        """
        return self.s

    def GetStringLength(self):
        r"""
    
        Obtain the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the current string.
    
        Example Output:
            "5"
        """
        return str(len(self.s))

    def UpdateCharIndex(self, char: str, index: int, char_index_dict: dict):
        r"""
    
        Update the character index dictionary to record the most recent occurrence position of the character.
    
        Args:
            char (str): The character to be updated.
            index (int): The position index where the character occurs.
            char_index_dict (dict): The current character index dictionary.
    
        Returns:
            str: The JSON string of the updated character index dictionary.
    
        Example Output:
            "{\"e\": 0}"
        """
        new_dict = char_index_dict.copy()
        new_dict[char] = index
        return json.dumps(new_dict)

    def CheckCharCount(self, char_index_dict: dict):
        r"""
    
        Check if the number of characters contained in the character index dictionary exceeds 2.
    
        Args:
            char_index_dict (dict): The current character index dictionary.
    
        Returns:
            str: Return "True" if the number of characters exceeds 2, otherwise return "False".
    
        Example Output:
            "True"
        """
        return str(len(char_index_dict) > 2)

    def AdjustLeftPointer(self, char_index_dict: dict, left: int):
        r"""
    
        Adjust the left pointer position when the number of characters exceeds 2.
    
        Args:
            char_index_dict (dict): The current character index dictionary.
            left (int): The current position of the left pointer.
    
        Returns:
            dict: A dictionary containing the adjusted left pointer and the updated character index dictionary.
    
        Example Output:
            {"left": 2, "char_index_dict": {"c": 1, "e": 2}}
        """
        new_dict = char_index_dict.copy()
        left_most_index = min(new_dict.values())
        del new_dict[self.s[left_most_index]]
        new_left = left_most_index + 1
        return {"left": new_left, "char_index_dict": new_dict}

    def CalculateCurrentLength(self, left: int, right: int):
        r"""
    
        Calculate the length of the current window.
    
        Args:
            left (int): The position of the left pointer.
            right (int): The position of the right pointer.
    
        Returns:
            str: The length of the current window.
    
        Example Output:
            "3"
        """
        return str(right - left + 1)

    def UpdateMaxLength(self, current_length: int, max_length: int):
        r"""
    
        Update the maximum window length.
    
        Args:
            current_length (int): The current window length.
            max_length (int): The current maximum window length.
    
        Returns:
            str: The updated maximum window length.
    
        Example Output:
            "5"
        """
        return str(max(max_length, current_length))

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
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
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        str_len = int(self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1])
        
        max_length = 0
        left = 0
        char_index_dict = {}
        
        for right in range(str_len):
            current_char = s[right]
            char_index_json = self.step(json.dumps({
                'name': 'UpdateCharIndex',
                'parameters': {
                    'char': current_char,
                    'index': right,
                    'char_index_dict': char_index_dict
                }
            }))[1]
            char_index_dict = json.loads(char_index_json)
            
            over_limit = self.step(json.dumps({
                'name': 'CheckCharCount',
                'parameters': {'char_index_dict': char_index_dict}
            }))[1]
            
            if over_limit == "True":
                adjust_result = self.step(json.dumps({
                    'name': 'AdjustLeftPointer',
                    'parameters': {
                        'char_index_dict': char_index_dict,
                        'left': left
                    }
                }))[1]
                adjust_result = json.loads(adjust_result)
                left = adjust_result['left']
                char_index_dict = adjust_result['char_index_dict']
            
            current_length = int(self.step(json.dumps({
                'name': 'CalculateCurrentLength',
                'parameters': {'left': left, 'right': right}
            }))[1])
            
            max_length = int(self.step(json.dumps({
                'name': 'UpdateMaxLength',
                'parameters': {
                    'current_length': current_length,
                    'max_length': max_length
                }
            }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "eceba"
    env = LongestTwoCharSubstringEnv.from_env_str(f"LongestTwoCharSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "ccaabbb"
    env = LongestTwoCharSubstringEnv.from_env_str(f"LongestTwoCharSubstringEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)