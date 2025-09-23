import gymnasium
import ast
import json

class PalindromePairCounterEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_PALINDROME = 0
        self.CONCATENATE_STRINGS = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckPalindrome": self.CHECK_PALINDROME,
            "ConcatenateStrings": self.CONCATENATE_STRINGS,
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
        prefix = "PalindromePairCounterEnv@"
        if not env_str.startswith(prefix):
            return None
        return PalindromePairCounterEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.strings = options.get("strings", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        count = 0
        n = len(self.strings)
        for i in range(n):
            for j in range(i + 1, n):
                if self.strings[i] + self.strings[j] == (self.strings[i] + self.strings[j])[::-1]:
                    count += 1
        return count

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
            
            if action_code == self.CHECK_PALINDROME:
                if "s" in params:
                    s = params["s"]
                    msg = self.CheckPalindrome(s)
                else:
                    msg = "Error: 's' parameter is missing for CHECK_PALINDROME action."
            
            elif action_code == self.CONCATENATE_STRINGS:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.ConcatenateStrings(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for CONCATENATE_STRINGS action."
                    
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
    def CheckPalindrome(self, s: str):
        r"""
    
        Check if the string s is a palindrome.
    
        Args:
            s (str): The string to be checked.
    
        Returns:
            str: "True" if it is a palindrome, "False" otherwise.
    
        Example Output:
            "True"
        """
        return str(s == s[::-1])

    def ConcatenateStrings(self, i: int, j: int):
        r"""
    
        Concatenate the i-th and j-th strings (indices start from 0).
    
        Args:
            i (int): The index of the first string.
            j (int): The index of the second string.
    
        Returns:
            str: The concatenated string.
    
        Example Output:
            "abccba"
        """
        if 0 <= i < len(self.strings) and 0 <= j < len(self.strings):
            return self.strings[i] + self.strings[j]
        return "Error: Invalid indices"

    def Observe(self):
        r"""
    
        Return the list of strings in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of strings in the environment.
    
        Example Output:
            "[\"abc\", \"cba\", \"bca\"]"
        """
        return json.dumps(self.strings)

    def Done(self, answer: int):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the number of string pairs that meet the conditions.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 1, Reference answer: 1, Result: Correct, reward=1"
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
        str_list_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        str_list = ast.literal_eval(str_list_str)
        n = len(str_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                concat_str = self.step(json.dumps({'name': 'ConcatenateStrings', 'parameters': {'i': i, 'j': j}}))[1]
                is_palindrome = self.step(json.dumps({'name': 'CheckPalindrome', 'parameters': {'s': concat_str}}))[1]
                if is_palindrome == "True":
                    count += 1
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (from first example)
    print("Test Case 1:")
    test_strings1 = ["abc", "cba", "bca"]
    env1 = PalindromePairCounterEnv.from_env_str(f"PalindromePairCounterEnv@{{\"strings\": {test_strings1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2 (from second example)
    print("\nTest Case 2:")
    test_strings2 = ["aa", "bb", "cc", "dd"]
    env2 = PalindromePairCounterEnv.from_env_str(f"PalindromePairCounterEnv@{{\"strings\": {test_strings2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)