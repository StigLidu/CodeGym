import gymnasium
import ast
import json
from collections import defaultdict

class ShortestSubstringEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.DONE = 1
        self.GET_STRING_LENGTH = 2
        self.GET_CHARACTER_AT_POSITION = 3
        self.COUNT_DISTINCT_IN_RANGE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "Done": self.DONE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "GetCharacterAtPosition": self.GET_CHARACTER_AT_POSITION,
            "CountDistinctInRange": self.COUNT_DISTINCT_IN_RANGE
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
        prefix = "ShortestSubstringEnv@"
        if not env_str.startswith(prefix):
            return None
        return ShortestSubstringEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.k = options.get("k", 1)
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
        n = len(self.s)
        if self.k > 26 or self.k > n:
            return -1
        
        char_count = defaultdict(int)
        start = 0
        min_length = float('inf')

        for end in range(n):
            char_count[self.s[end]] += 1

            while len(char_count) >= self.k:
                current_length = end - start + 1
                if current_length < min_length:
                    min_length = current_length

                char_count[self.s[start]] -= 1
                if char_count[self.s[start]] == 0:
                    del char_count[self.s[start]]
                start += 1
                
        return min_length if min_length != float('inf') else -1

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
            elif action_code == self.GET_CHARACTER_AT_POSITION:
                if "pos" in params:
                    pos = params["pos"]
                    msg = self.GetCharacterAtPosition(pos)
                else:
                    msg = "Error: 'pos' parameter is missing for GET_CHARACTER_AT_POSITION action."
            elif action_code == self.COUNT_DISTINCT_IN_RANGE:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CountDistinctInRange(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for COUNT_DISTINCT_IN_RANGE action."
        except Exception as e:
            msg = f"Error: {str(e)}"

        return True, msg

    # All the actions of the environment
    def Observe(self):
        r"""
    
        Obtain the k value and string in the current environment.
    
        Args:
            None
    
        Returns:
            str: Information about the current k value and string.
    
        Example Output:
            "k=3, s=abcabcabc"
        """
        return f"k={self.k}, s={self.s}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user, which is the length of the shortest substring or -1.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
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
    
        Obtain the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the string.
    
        Example Output:
            "9"
        """
        return str(len(self.s))

    def GetCharacterAtPosition(self, pos: int):
        r"""
    
        Obtain the character at the specified position in the string.
    
        Args:
            pos (int): The position index of the character to be obtained (starting from 0).
    
        Returns:
            str: The character at the specified position; returns an error message if the index is invalid.
    
        Example Output:
            "a"
        """
        if 0 <= pos < len(self.s):
            return self.s[pos]
        else:
            return "Error: Invalid position"

    def CountDistinctInRange(self, start: int, end: int):
        r"""
    
        Calculate the number of distinct characters in the range from start to end (inclusive) in the string.
    
        Args:
            start (int): The starting position index (starting from 0).
            end (int): The ending position index (starting from 0).
    
        Returns:
            str: The number of distinct characters; returns an error message if the range is invalid.
    
        Example Output:
            "3"
        """
        if start < 0 or end >= len(self.s) or start > end:
            return "Error: Invalid range"
        substring = self.s[start:end+1]
        return str(len(set(substring)))

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        import json
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        k = int(observe_result.split(',')[0].split('=')[1].strip())
        str_len_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        str_len = int(str_len_str)
        
        if k <= 0:
            min_length = -1
        else:
            min_length = float('inf')
            for start in range(str_len):
                for end in range(start, str_len):
                    distinct_count_str = self.step(json.dumps({
                        'name': 'CountDistinctInRange',
                        'parameters': {'start': start, 'end': end}
                    }))[1]
                    distinct_count = int(distinct_count_str)
                    if distinct_count >= k:
                        current_length = end - start + 1
                        if current_length < min_length:
                            min_length = current_length
            if min_length == float('inf'):
                min_length = -1
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': min_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env1 = ShortestSubstringEnv.from_env_str('ShortestSubstringEnv@{"k": 3, "s": "abcabcabc"}')
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    env2 = ShortestSubstringEnv.from_env_str('ShortestSubstringEnv@{"k": 5, "s": "aabbcc"}')
    print(env2.solve())
    print("step count:", env2.step_count)

    # test case 3
    print("\nTest Case 3:")
    env3 = ShortestSubstringEnv.from_env_str('ShortestSubstringEnv@{"k": 2, "s": "aaaaabbbbb"}')
    print(env3.solve())
    print("step count:", env3.step_count)