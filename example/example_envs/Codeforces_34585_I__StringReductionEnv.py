import gymnasium
import ast
import json

class StringReductionEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.IDENTIFY_SAME_CHAR_SUBSTRINGS = 1
        self.REMOVE_SUBSTRING = 2
        self.CHECK_IF_EMPTY = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "IdentifySameCharSubstrings": self.IDENTIFY_SAME_CHAR_SUBSTRINGS,
            "RemoveSubstring": self.REMOVE_SUBSTRING,
            "CheckIfEmpty": self.CHECK_IF_EMPTY,
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
        prefix = "StringReductionEnv@"
        if not env_str.startswith(prefix):
            return None
        return StringReductionEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.original_string = options.get("string", "")
        self.current_string = self.original_string
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        return 0

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
            
            elif action_code == self.IDENTIFY_SAME_CHAR_SUBSTRINGS:
                if "string" in params:
                    string = params["string"]
                    msg = self.IdentifySameCharSubstrings(string)
                else:
                    msg = "Error: 'string' parameter is missing for IDENTIFY_SAME_CHAR_SUBSTRINGS action."
                    
            elif action_code == self.REMOVE_SUBSTRING:
                if "string" in params and "start" in params and "end" in params:
                    string = params["string"]
                    start = params["start"]
                    end = params["end"]
                    msg = self.RemoveSubstring(string, start, end)
                else:
                    msg = "Error: 'string', 'start' or 'end' parameter is missing for REMOVE_SUBSTRING action."
                    
            elif action_code == self.CHECK_IF_EMPTY:
                if "string" in params:
                    string = params["string"]
                    msg = self.CheckIfEmpty(string)
                else:
                    msg = "Error: 'string' parameter is missing for CHECK_IF_EMPTY action."
                    
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
            "aabbcc"
        """
        return self.current_string

    def IdentifySameCharSubstrings(self, string: str):
        r"""
    
        Identify all consecutive substrings composed of the same character in the string.
    
        Args:
            string (str): The string to be identified.
    
        Returns:
            str: A list containing information of all same-character substrings, where each element is a dictionary including the character, start index, and end index.
    
        Example Output:
            "[{\"char\":\"a\",\"start\":0,\"end\":1},{\"char\":\"b\",\"start\":2,\"end\":3},{\"char\":\"c\",\"start\":4,\"end\":5}]"
        """
        if not string:
            return json.dumps([])
            
        substrings = []
        current_char = string[0]
        start_index = 0
        
        for i in range(1, len(string)):
            if string[i] != current_char:
                substrings.append({
                    "char": current_char,
                    "start": start_index,
                    "end": i - 1
                })
                current_char = string[i]
                start_index = i
        
        substrings.append({
            "char": current_char,
            "start": start_index,
            "end": len(string) - 1
        })
        
        return json.dumps(substrings)

    def RemoveSubstring(self, string: str, start: int, end: int):
        r"""
    
        Remove the substring within the specified range from the string.
    
        Args:
            string (str): The original string.
            start (int): The start index of the substring (inclusive).
            end (int): The end index of the substring (inclusive).
    
        Returns:
            str: The new string after removing the substring.
    
        Example Output:
            "bbcc"
        """
        self.current_string = string[:start] + string[end+1:]
        return self.current_string

    def CheckIfEmpty(self, string: str):
        r"""
    
        Check if the string is empty.
    
        Args:
            string (str): The string to be checked.
    
        Returns:
            str: Returns "True" if empty, otherwise returns "False".
    
        Example Output:
            "True"
        """
        return str(string == "").lower()

    def Done(self, answer: int):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The final string length submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 0, Reference answer: 0, Result: Correct, reward=1"
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
        current_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        
        while True:
            is_empty = self.step(json.dumps({'name': 'CheckIfEmpty', 'parameters': {'string': current_str}}))[1]
            if is_empty == "True":
                break
            
            substrings_str = self.step(json.dumps({'name': 'IdentifySameCharSubstrings', 'parameters': {'string': current_str}}))[1]
            substrings = ast.literal_eval(substrings_str)
            
            if not substrings:
                break
            
            first_sub = substrings[0]
            start = first_sub['start']
            end = first_sub['end']
            
            current_str = self.step(json.dumps({'name': 'RemoveSubstring', 'parameters': {'string': current_str, 'start': start, 'end': end}}))[1]
        
        final_length = len(current_str)
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "aabbcc"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "abcddcba"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_string = "aaaaaaa"
    env = StringReductionEnv.from_env_str(f"StringReductionEnv@{{\"string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)