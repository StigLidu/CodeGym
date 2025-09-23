import gymnasium
import ast
import json
from collections import defaultdict

class LargestEvenPalindromeEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.COUNT_CHARACTER = 0
        self.BUILD_HALF_PALINDROME = 1
        self.REVERSE_STRING = 2
        self.CONCATENATE_STRINGS = 3
        self.CHECK_EVEN_LENGTH = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "CountCharacter": self.COUNT_CHARACTER,
            "BuildHalfPalindrome": self.BUILD_HALF_PALINDROME,
            "ReverseString": self.REVERSE_STRING,
            "ConcatenateStrings": self.CONCATENATE_STRINGS,
            "CheckEvenLength": self.CHECK_EVEN_LENGTH,
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
        prefix = "LargestEvenPalindromeEnv@"
        if not env_str.startswith(prefix):
            return None
        return LargestEvenPalindromeEnv(env_str=env_str)

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
        char_count = defaultdict(int)
        for char in self.s:
            char_count[char] += 1
        
        half_palindrome = ""
        for char, count in char_count.items():
            half_palindrome += char * (count // 2)
        
        full_palindrome = half_palindrome + half_palindrome[::-1]
        
        return full_palindrome if len(full_palindrome) % 2 == 0 else ""

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
            
            if action_code == self.COUNT_CHARACTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.CountCharacter(char)
                else:
                    msg = "Error: 'char' parameter is missing for COUNT_CHARACTER action."
            
            elif action_code == self.BUILD_HALF_PALINDROME:
                if "char_counts" in params:
                    char_counts = params["char_counts"]
                    msg = self.BuildHalfPalindrome(char_counts)
                else:
                    msg = "Error: 'char_counts' parameter is missing for BUILD_HALF_PALINDROME action."
            
            elif action_code == self.REVERSE_STRING:
                if "s" in params:
                    s = params["s"]
                    msg = self.ReverseString(s)
                else:
                    msg = "Error: 's' parameter is missing for REVERSE_STRING action."
            
            elif action_code == self.CONCATENATE_STRINGS:
                if "s1" in params and "s2" in params:
                    s1 = params["s1"]
                    s2 = params["s2"]
                    msg = self.ConcatenateStrings(s1, s2)
                else:
                    msg = "Error: 's1' or 's2' parameter is missing for CONCATENATE_STRINGS action."
            
            elif action_code == self.CHECK_EVEN_LENGTH:
                if "s" in params:
                    s = params["s"]
                    msg = self.CheckEvenLength(s)
                else:
                    msg = "Error: 's' parameter is missing for CHECK_EVEN_LENGTH action."
            
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
    def CountCharacter(self, char: str):
        r"""
    
        Count the number of occurrences of a specific character in a string.
    
        Args:
            char (str): The single character to be counted.
    
        Returns:
            str: The number of occurrences of the character.
    
        Example Output:
            "3"
        """
        count = self.s.count(char)
        return str(count)

    def BuildHalfPalindrome(self, char_counts: dict):
        r"""
    
        Build half of a palindrome based on character counts.
    
        Args:
            char_counts (dict): A mapping dictionary from characters to their counts.
    
        Returns:
            str: The constructed half-palindrome string.
    
        Example Output:
            "aabb"
        """
        half_palindrome = ""
        for char, count in char_counts.items():
            half_palindrome += char * (count // 2)
        return half_palindrome

    def ReverseString(self, s: str):
        r"""
    
        Reverse the input string.
    
        Args:
            s (str): The string to be reversed.
    
        Returns:
            str: The reversed string.
    
        Example Output:
            "cba"
        """
        return s[::-1]

    def ConcatenateStrings(self, s1: str, s2: str):
        r"""
    
        Concatenate two strings together.
    
        Args:
            s1 (str): The first string.
            s2 (str): The second string.
    
        Returns:
            str: The concatenated string.
    
        Example Output:
            "abcdef"
        """
        return s1 + s2

    def CheckEvenLength(self, s: str):
        r"""
    
        Check if the length of the string is even.
    
        Args:
            s (str): The string to be checked.
    
        Returns:
            str: "True" indicates the length is even, "False" indicates the length is odd.
    
        Example Output:
            "True"
        """
        return str(len(s) % 2 == 0)

    def Observe(self):
        r"""
    
        Return the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "abbaabc"
        """
        return self.s

    def Done(self, answer: str):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: abba, Reference answer: abba, Result: Correct, reward=1"
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
        current_string = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        
        char_counts = {}
        for char in 'abcdefghijklmnopqrstuvwxyz':
            count_str = self.step(json.dumps({'name': 'CountCharacter', 'parameters': {'char': char}}))[1]
            char_counts[char] = int(count_str)
        
        half_pal = self.step(json.dumps({'name': 'BuildHalfPalindrome', 'parameters': {'char_counts': char_counts}}))[1]
        
        reversed_half = self.step(json.dumps({'name': 'ReverseString', 'parameters': {'s': half_pal}}))[1]
        
        full_palindrome = self.step(json.dumps({'name': 'ConcatenateStrings', 'parameters': {'s1': half_pal, 's2': reversed_half}}))[1]
        
        is_even = self.step(json.dumps({'name': 'CheckEvenLength', 'parameters': {'s': full_palindrome}}))[1]
        
        if not full_palindrome or is_even == "False":
            answer = ""
        else:
            answer = full_palindrome
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "abbaabc"
    env = LargestEvenPalindromeEnv.from_env_str(f"LargestEvenPalindromeEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "abcde"
    env = LargestEvenPalindromeEnv.from_env_str(f"LargestEvenPalindromeEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)