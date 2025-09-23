import gymnasium
import json
import ast

class MaxConsecutiveOnesEnv(gymnasium.Env):
    def __init__(self, env_str: str = None):
        super().__init__()
        
        # [Required] Define the action names
        self.FLIP_SUBSTRING = 0
        self.COUNT_CONSECUTIVE_ONES = 1
        self.RESTORE_ORIGINAL = 2
        self.OBSERVE = 3
        self.DONE = 4
        
        # [Required] Define the action mapping
        self.func_mapping = {
            "FlipSubstring": self.FLIP_SUBSTRING,
            "CountConsecutiveOnes": self.COUNT_CONSECUTIVE_ONES,
            "RestoreOriginal": self.RESTORE_ORIGINAL,
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
        prefix = "MaxConsecutiveOnesEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxConsecutiveOnesEnv(env_str=env_str)
    
    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.original_s = options.get("s", "")
        self.current_s = self.original_s
        self.flip_performed = False
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."
    
    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        n = len(self.original_s)
        total_ones = sum(int(char) for char in self.original_s)
        
        if total_ones == n:
            return n
        
        max_1s = 0
        
        for i in range(n):
            for j in range(i, n):
                flipped = self.original_s[:i] + ''.join('1' if x == '0' else '0' for x in self.original_s[i:j+1]) + self.original_s[j+1:]
                max_1s = max(max_1s, max(len(x) for x in flipped.split('0')))
        
        return max_1s
    
    # [Required] Define the step method of the environment
    def step(self, action: str):
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
            
            if action_code == self.FLIP_SUBSTRING:
                if "start" in params and "end" in params and not self.flip_performed:
                    start = params["start"]
                    end = params["end"]
                    msg = self.FlipSubstring(start, end)
                elif self.flip_performed:
                    msg = "Error: Flip operation can only be performed once."
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for FLIP_SUBSTRING action."
            
            elif action_code == self.COUNT_CONSECUTIVE_ONES:
                msg = self.CountConsecutiveOnes()
            
            elif action_code == self.RESTORE_ORIGINAL:
                msg = self.RestoreOriginal()
            
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
    def FlipSubstring(self, start: int, end: int):
        r"""
    
        Flip the substring from start to end (inclusive), changing 0 to 1 and 1 to 0. This operation can only be performed once.
        
        Args:
            start (int): The starting index of the substring (0-based)
            end (int): The ending index of the substring (0-based)
            
        Returns:
            str: The string after the flip operation
            
        Example Output:
            "111001"
        """
        if start < 0 or end >= len(self.current_s) or start > end:
            return "Error: Invalid start or end index."
            
        prefix = self.current_s[:start]
        flipped = ''.join('1' if c == '0' else '0' for c in self.current_s[start:end+1])
        suffix = self.current_s[end+1:]
        self.current_s = prefix + flipped + suffix
        self.flip_performed = True
        
        return self.current_s
    
    def CountConsecutiveOnes(self):
        r"""
    
        Calculate the maximum number of consecutive 1s in the current string.
        
        Args:
            None
            
        Returns:
            str: The maximum number of consecutive 1s in the current string
            
        Example Output:
            "5"
        """
        max_count = 0
        current_count = 0
        
        for c in self.current_s:
            if c == '1':
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return str(max_count)
    
    def RestoreOriginal(self):
        r"""
    
        Restore the current string to the original string and reset the flip operation state.
        
        Args:
            None
            
        Returns:
            str: The restored original string
            
        Example Output:
            "1101101"
        """
        self.current_s = self.original_s
        self.flip_performed = False
        return self.current_s
    
    def Observe(self):
        r"""
    
        Get the current binary string.
        
        Args:
            None
            
        Returns:
            str: The current binary string
            
        Example Output:
            "1101101"
        """
        return self.current_s
    
    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
        
        Args:
            answer (int): The answer submitted by the user, i.e., the maximum number of consecutive 1s.
            
        Returns:
            str: Result information, including whether it is correct and reward information.
            
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
        original = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = len(original)
        max_ones = 0
        
        current_count = int(self.step(json.dumps({'name': 'CountConsecutiveOnes', 'parameters': {}}))[1])
        max_ones = current_count
        
        for start in range(n):
            for end in range(start, n):
                self.step(json.dumps({'name': 'FlipSubstring', 'parameters': {'start': start, 'end': end}}))
                current = int(self.step(json.dumps({'name': 'CountConsecutiveOnes', 'parameters': {}}))[1])
                if current > max_ones:
                    max_ones = current
                self.step(json.dumps({'name': 'RestoreOriginal', 'parameters': {}}))
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_ones}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "1101101"
    env = MaxConsecutiveOnesEnv.from_env_str(f"MaxConsecutiveOnesEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 2
    print("\nTest Case 2:")
    test_str = "0000"
    env = MaxConsecutiveOnesEnv.from_env_str(f"MaxConsecutiveOnesEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)