import gymnasium
import ast
import json

class PalindromeCounterEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_PALINDROME = 0
        self.ADD_TO_SET = 1
        self.GET_SET_SIZE = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckPalindrome": self.CHECK_PALINDROME,
            "AddToSet": self.ADD_TO_SET,
            "GetSetSize": self.GET_SET_SIZE,
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
        prefix = "PalindromeCounterEnv@"
        if not env_str.startswith(prefix):
            return None
        return PalindromeCounterEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.palindromic_substrings = set()
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
        dp = [[False] * n for _ in range(n)]
        palindromic_substrings = set()

        for length in range(1, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                if length == 1:
                    dp[start][end] = True
                elif length == 2:
                    dp[start][end] = (self.s[start] == self.s[end])
                else:
                    dp[start][end] = (self.s[start] == self.s[end]) and dp[start + 1][end - 1]

                if dp[start][end]:
                    palindromic_substrings.add(self.s[start:end + 1])

        return len(palindromic_substrings)

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
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CheckPalindrome(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for CHECK_PALINDROME action."
            
            elif action_code == self.ADD_TO_SET:
                if "substring" in params:
                    substring = params["substring"]
                    msg = self.AddToSet(substring)
                else:
                    msg = "Error: 'substring' parameter is missing for ADD_TO_SET action."
                    
            elif action_code == self.GET_SET_SIZE:
                msg = self.GetSetSize()
                
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
    def CheckPalindrome(self, start: int, end: int):
        r"""
    
        Check if the substring from start to end (inclusive) is a palindrome.
    
        Args:
            start (int): Starting index of the substring
            end (int): Ending index of the substring
    
        Returns:
            str: "True" indicates it is a palindrome, "False" indicates it is not
    
        Example Output:
            "True"
        """
        if start < 0 or end >= len(self.s) or start > end:
            return "False"
            
        substring = self.s[start:end+1]
        return str(substring == substring[::-1])

    def AddToSet(self, substring: str):
        r"""
    
        Add the substring to the palindromic substring set (automatically removes duplicates).
    
        Args:
            substring (str): The substring to be added
    
        Returns:
            str: The number of elements in the current set
    
        Example Output:
            "3"
        """
        self.palindromic_substrings.add(substring)
        return str(len(self.palindromic_substrings))

    def GetSetSize(self):
        r"""
    
        Get the size of the palindromic substring set.
    
        Args:
            None
    
        Returns:
            str: The number of elements in the set
    
        Example Output:
            "5"
        """
        return str(len(self.palindromic_substrings))

    def Observe(self):
        r"""
    
        Get the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment
    
        Example Output:
            "ababa"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user
    
        Returns:
            str: Result information, including correctness and reward details
    
        Example Output:
            "Your answer: 5, Reference answer: 5, Result: Correct, reward=1"
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
        
        for start in range(n):
            for end in range(start, n):
                is_palindrome = self.step(json.dumps({
                    'name': 'CheckPalindrome',
                    'parameters': {'start': start, 'end': end}
                }))[1]
                if is_palindrome == "True":
                    substring = s[start:end+1]
                    self.step(json.dumps({
                        'name': 'AddToSet',
                        'parameters': {'substring': substring}
                    }))
        
        answer = int(self.step(json.dumps({'name': 'GetSetSize', 'parameters': {}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "ababa"
    env = PalindromeCounterEnv.from_env_str(f"PalindromeCounterEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "abcba"
    env = PalindromeCounterEnv.from_env_str(f"PalindromeCounterEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)