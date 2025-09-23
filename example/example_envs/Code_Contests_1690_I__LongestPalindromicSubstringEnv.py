import gymnasium
import json
import ast

class LongestPalindromicSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_EVEN_PALINDROME = 1
        self.CHECK_ODD_PALINDROME = 2
        self.UPDATE_MAX_PALINDROME = 3
        self.GET_LONGEST_PALINDROME = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckEvenPalindrome": self.CHECK_EVEN_PALINDROME,
            "CheckOddPalindrome": self.CHECK_ODD_PALINDROME,
            "UpdateMaxPalindrome": self.UPDATE_MAX_PALINDROME,
            "GetLongestPalindrome": self.GET_LONGEST_PALINDROME,
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
        prefix = "LongestPalindromicSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestPalindromicSubstringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.start = 0
        self.max_length = 1 if len(self.s) > 0 else 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.s)
        if n == 0:
            return ""
        
        start = 0
        max_length = 1
        
        for i in range(1, n):
            low = i - 1
            high = i
            while low >= 0 and high < n and self.s[low] == self.s[high]:
                if high - low + 1 > max_length:
                    start = low
                    max_length = high - low + 1
                low -= 1
                high += 1
            
            low = i - 1
            high = i + 1
            while low >= 0 and high < n and self.s[low] == self.s[high]:
                if high - low + 1 > max_length:
                    start = low
                    max_length = high - low + 1
                low -= 1
                high += 1

        return self.s[start:start + max_length]

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
            
            elif action_code == self.CHECK_EVEN_PALINDROME:
                if "i" in params:
                    i = params["i"]
                    msg = self.CheckEvenPalindrome(i)
                else:
                    msg = "Error: 'i' parameter is missing for CHECK_EVEN_PALINDROME action."
            
            elif action_code == self.CHECK_ODD_PALINDROME:
                if "i" in params:
                    i = params["i"]
                    msg = self.CheckOddPalindrome(i)
                else:
                    msg = "Error: 'i' parameter is missing for CHECK_ODD_PALINDROME action."
            
            elif action_code == self.UPDATE_MAX_PALINDROME:
                if "start" in params and "length" in params:
                    start = params["start"]
                    length = params["length"]
                    msg = self.UpdateMaxPalindrome(start, length)
                else:
                    msg = "Error: 'start' or 'length' parameter is missing for UPDATE_MAX_PALINDROME action."
            
            elif action_code == self.GET_LONGEST_PALINDROME:
                msg = self.GetLongestPalindrome()
            
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
            "babad"
        """
        return self.s

    def CheckEvenPalindrome(self, i: int):
        r"""
    
        Check for even-length palindromes centered at position i.
    
        Args:
            i (int): The index of the center position to check.
    
        Returns:
            str: A JSON string containing the starting position and length of the longest even-length palindrome.
    
        Example Output:
            "{\"start\": 1, \"length\": 2}"
        """
        n = len(self.s)
        low = i - 1
        high = i
        current_start = low
        current_length = 0
        
        while low >= 0 and high < n and self.s[low] == self.s[high]:
            current_start = low
            current_length = high - low + 1
            low -= 1
            high += 1
            
        return json.dumps({"start": current_start, "length": current_length})

    def CheckOddPalindrome(self, i: int):
        r"""
    
        Check for odd-length palindromes centered at position i.
    
        Args:
            i (int): The index of the center position to check.
    
        Returns:
            str: A JSON string containing the starting position and length of the longest odd-length palindrome.
    
        Example Output:
            "{\"start\": 0, \"length\": 3}"
        """
        n = len(self.s)
        low = i - 1
        high = i + 1
        current_start = i
        current_length = 1
        
        while low >= 0 and high < n and self.s[low] == self.s[high]:
            current_start = low
            current_length = high - low + 1
            low -= 1
            high += 1
            
        return json.dumps({"start": current_start, "length": current_length})

    def UpdateMaxPalindrome(self, start: int, length: int):
        r"""
    
        Update the starting position and length of the longest palindrome.
    
        Args:
            start (int): The starting position of the newly discovered palindrome.
            length (int): The length of the newly discovered palindrome.
    
        Returns:
            str: The updated length of the longest palindrome.
    
        Example Output:
            "3"
        """
        if length > self.max_length:
            self.start = start
            self.max_length = length
        return str(self.max_length)

    def GetLongestPalindrome(self):
        r"""
    
        Obtain the longest palindromic substring based on the recorded starting position and length.
    
        Args:
            None
    
        Returns:
            str: The longest palindromic substring.
    
        Example Output:
            "bab"
        """
        if self.max_length == 0:
            return ""
        return self.s[self.start:self.start + self.max_length]

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: bab, Reference answer: bab, Result: Correct, reward=1"
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
        n = len(s)
        
        if n == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': ""}}))[1]
        
        for i in range(n):
            odd_result = self.step(json.dumps({'name': 'CheckOddPalindrome', 'parameters': {'i': i}}))[1]
            odd_data = json.loads(odd_result)
            self.step(json.dumps({'name': 'UpdateMaxPalindrome', 'parameters': {'start': odd_data['start'], 'length': odd_data['length']}}))
            
            even_result = self.step(json.dumps({'name': 'CheckEvenPalindrome', 'parameters': {'i': i}}))[1]
            even_data = json.loads(even_result)
            self.step(json.dumps({'name': 'UpdateMaxPalindrome', 'parameters': {'start': even_data['start'], 'length': even_data['length']}}))
        
        longest_palindrome = self.step(json.dumps({'name': 'GetLongestPalindrome', 'parameters': {}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': longest_palindrome}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "babad"
    env = LongestPalindromicSubstringEnv.from_env_str(
        f"LongestPalindromicSubstringEnv@{{\"s\": \"{test_str}\"}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "cbbd"
    env = LongestPalindromicSubstringEnv.from_env_str(
        f"LongestPalindromicSubstringEnv@{{\"s\": \"{test_str}\"}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 3
    print("\nTest Case 3:")
    test_str = "a"
    env = LongestPalindromicSubstringEnv.from_env_str(
        f"LongestPalindromicSubstringEnv@{{\"s\": \"{test_str}\"}}"
    )
    print(env.solve())
    print("step count:", env.step_count)