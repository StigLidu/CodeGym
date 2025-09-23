import gymnasium
import ast
import json

class PalindromeSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.REMOVE_CHARACTER = 0
        self.CHECK_SUBSEQUENCE = 1
        self.CHECK_PALINDROME_POSSIBILITY = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "RemoveCharacterAtPosition": self.REMOVE_CHARACTER,
            "CheckIsSubsequence": self.CHECK_SUBSEQUENCE,
            "CheckCanBePalindrome": self.CHECK_PALINDROME_POSSIBILITY,
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
        prefix = "PalindromeSubsequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        return PalindromeSubsequenceEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.t = options.get("t", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def can_be_palindrome(s):
            freq = {}
            for ch in s:
                freq[ch] = freq.get(ch, 0) + 1
            odd_count = sum(1 for count in freq.values() if count % 2 != 0)
            return odd_count <= 1

        def is_subsequence(s, t):
            it = iter(t)
            return all(c in it for c in s)

        for i in range(len(self.s)):
            new_string = self.s[:i] + self.s[i + 1:]
            if is_subsequence(new_string, self.t) and can_be_palindrome(new_string):
                return True

        return False

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
            
            if action_code == self.REMOVE_CHARACTER:
                if "s" in params and "position" in params:
                    s = params["s"]
                    position = params["position"]
                    msg = self.RemoveCharacterAtPosition(s, position)
                else:
                    msg = "Error: 's' or 'position' parameter is missing for REMOVE_CHARACTER action."
            
            elif action_code == self.CHECK_SUBSEQUENCE:
                if "s" in params and "t" in params:
                    s = params["s"]
                    t = params["t"]
                    msg = self.CheckIsSubsequence(s, t)
                else:
                    msg = "Error: 's' or 't' parameter is missing for CHECK_SUBSEQUENCE action."
                    
            elif action_code == self.CHECK_PALINDROME_POSSIBILITY:
                if "s" in params:
                    s = params["s"]
                    msg = self.CheckCanBePalindrome(s)
                else:
                    msg = "Error: 's' parameter is missing for CHECK_PALINDROME_POSSIBILITY action."
                    
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
    def RemoveCharacterAtPosition(self, s: str, position: int):
        r"""
    
        Remove the character at the specified position from the string s.
    
        Args:
            s (str): Original string
            position (int): The position of the character to be removed
    
        Returns:
            str: The new string after removing the character
    
        Example Output:
            "abc"
        """
        if position < 0 or position >= len(s):
            return ""
        return s[:position] + s[position+1:]

    def CheckIsSubsequence(self, s: str, t: str):
        r"""
    
        Check if string s is a subsequence of t.
    
        Args:
            s (str): The string to be checked
            t (str): The target string
    
        Returns:
            str: "true" indicates it is a subsequence, "false" indicates it is not
    
        Example Output:
            "true"
        """
        it = iter(t)
        result = all(c in it for c in s)
        return "true" if result else "false"

    def CheckCanBePalindrome(self, s: str):
        r"""
    
        Check if the string s can be rearranged to form a palindrome.
    
        Args:
            s (str): The string to be checked
    
        Returns:
            str: "true" indicates it can form a palindrome, "false" indicates it cannot
    
        Example Output:
            "true"
        """
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        odd_count = sum(1 for count in freq.values() if count % 2 != 0)
        return "true" if odd_count <= 1 else "false"

    def Observe(self):
        r"""
    
        Obtain the strings s and t in the current environment.
    
        Args:
            None
    
        Returns:
            str: Information containing the current s and t
    
        Example Output:
            "s: abcde, t: aebdc"
        """
        return f"s: {self.s}, t: {self.t}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (bool): The answer submitted by the user
    
        Returns:
            str: Result information, including whether it is correct and reward information
    
        Example Output:
            "Your answer: true, Reference answer: true, Result: Correct, reward=1"
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
        observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        s = observe_info.split('s: ')[1].split(', t: ')[0]
        t = observe_info.split('t: ')[1]
        
        n = len(s)
        for i in range(n):
            new_s = self.step(json.dumps({
                'name': 'RemoveCharacterAtPosition',
                'parameters': {'s': s, 'position': i}
            }))[1]
            
            is_subseq = self.step(json.dumps({
                'name': 'CheckIsSubsequence',
                'parameters': {'s': new_s, 't': t}
            }))[1] == "true"
            
            if is_subseq:
                can_palindrome = self.step(json.dumps({
                    'name': 'CheckCanBePalindrome',
                    'parameters': {'s': new_s}
                }))[1] == "true"
                
                if can_palindrome:
                    return self.step(json.dumps({
                        'name': 'Done',
                        'parameters': {'answer': True}
                    }))[1]
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': False}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env1 = PalindromeSubsequenceEnv.from_env_str('PalindromeSubsequenceEnv@{"s": "abcde", "t": "aebdc"}')
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    env2 = PalindromeSubsequenceEnv.from_env_str('PalindromeSubsequenceEnv@{"s": "abc", "t": "def"}')
    print(env2.solve())
    print("step count:", env2.step_count)