import gymnasium
import json
import ast

class LongestPalindromeEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_STRING_LENGTH = 1
        self.EXPAND_AROUND_CENTER = 2
        self.COMPARE_LENGTHS = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "ExpandAroundCenter": self.EXPAND_AROUND_CENTER,
            "CompareLengths": self.COMPARE_LENGTHS,
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
        prefix = "LongestPalindromeEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestPalindromeEnv(env_str=env_str)

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
        s = self.s
        n = len(s)
        if n == 0:
            return ""
        
        longest_palindrome = ""
        
        def expand_around_center(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]
        
        for i in range(n):
            odd_palindrome = expand_around_center(i, i)
            even_palindrome = expand_around_center(i, i + 1)
            
            current_longest = max(odd_palindrome, even_palindrome, key=len)
            if len(current_longest) > len(longest_palindrome):
                longest_palindrome = current_longest
        
        return longest_palindrome

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
            
            if action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
            elif action_code == self.EXPAND_AROUND_CENTER:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.ExpandAroundCenter(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for EXPAND_AROUND_CENTER action."
            elif action_code == self.COMPARE_LENGTHS:
                if "str1" in params and "str2" in params:
                    str1 = params["str1"]
                    str2 = params["str2"]
                    msg = self.CompareLengths(str1, str2)
                else:
                    msg = "Error: 'str1' or 'str2' parameter is missing for COMPARE_LENGTHS action."
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
    def Observe(self):
        r"""
    
        Returns the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "babad"
        """
        return self.s

    def GetStringLength(self):
        r"""
    
        Gets the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the current string.
    
        Example Output:
            "5"
        """
        return str(len(self.s))

    def ExpandAroundCenter(self, left: int, right: int):
        r"""
    
        Expands from the specified left and right centers to both sides to find the longest palindromic substring.
    
        Args:
            left (int): Left center index
            right (int): Right center index
    
        Returns:
            str: The longest palindromic substring found.
    
        Example Output:
            "bab"
        """
        s = self.s
        n = len(s)
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    def CompareLengths(self, str1: str, str2: str):
        r"""
    
        Compares the lengths of two strings and returns the longer one. If the lengths are the same, returns the first one.
    
        Args:
            str1 (str): The first string
            str2 (str): The second string
    
        Returns:
            str: The longer string.
    
        Example Output:
            "aba"
        """
        return str1 if len(str1) >= len(str2) else str2

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (str): The longest palindromic substring submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: bab, Reference answer: bab, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = len(answer) == len(ref_answer) and answer == ref_answer
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
        length_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        n = int(length_str)
        
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': ""}}))[1]
        
        longest = s[0] if n >= 1 else ""
        
        for i in range(n):
            odd_pal = self.step(json.dumps({
                'name': 'ExpandAroundCenter',
                'parameters': {'left': i, 'right': i}
            }))[1]
            longest = self.step(json.dumps({
                'name': 'CompareLengths',
                'parameters': {'str1': longest, 'str2': odd_pal}
            }))[1]
            
            if i < n - 1:
                even_pal = self.step(json.dumps({
                    'name': 'ExpandAroundCenter',
                    'parameters': {'left': i, 'right': i + 1}
                }))[1]
                longest = self.step(json.dumps({
                    'name': 'CompareLengths',
                    'parameters': {'str1': longest, 'str2': even_pal}
                }))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': longest}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "babad"
    env = LongestPalindromeEnv.from_env_str(f"LongestPalindromeEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "cbbd"
    env = LongestPalindromeEnv.from_env_str(f"LongestPalindromeEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)