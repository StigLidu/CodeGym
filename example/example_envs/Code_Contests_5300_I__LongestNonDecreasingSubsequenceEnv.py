import gymnasium
import json
import ast
import random

class LongestNonDecreasingSubsequenceEnv(gymnasium.Env):
    def __init__(self, env_str: str = None):
        super().__init__()
        
        # [Required] Define the action names
        self.INITIALIZE_DP_ARRAY = 0
        self.UPDATE_DP_VALUE = 1
        self.FIND_MAX_DP_VALUE = 2
        self.OBSERVE = 3
        self.DONE = 4
        
        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeDPArray": self.INITIALIZE_DP_ARRAY,
            "UpdateDPValue": self.UPDATE_DP_VALUE,
            "FindMaxDPValue": self.FIND_MAX_DP_VALUE,
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
        prefix = "LongestNonDecreasingSubsequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestNonDecreasingSubsequenceEnv(env_str=env_str)
    
    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.sequence = options.get("sequence", [])
        self.dp = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."
    
    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.sequence)
        if n == 0:
            return 0
        
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if self.sequence[i] >= self.sequence[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
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
            
            if action_code == self.INITIALIZE_DP_ARRAY:
                if "length" in params:
                    length = params["length"]
                    msg = self.InitializeDPArray(length)
                else:
                    msg = "Error: 'length' parameter is missing for INITIALIZE_DP_ARRAY action."
            
            elif action_code == self.UPDATE_DP_VALUE:
                if "i" in params and "j" in params and "sequence_i" in params and "sequence_j" in params:
                    i = params["i"]
                    j = params["j"]
                    sequence_i = params["sequence_i"]
                    sequence_j = params["sequence_j"]
                    msg = self.UpdateDPValue(i, j, sequence_i, sequence_j)
                else:
                    msg = "Error: 'i', 'j', 'sequence_i' or 'sequence_j' parameter is missing for UPDATE_DP_VALUE action."
            
            elif action_code == self.FIND_MAX_DP_VALUE:
                msg = self.FindMaxDPValue()
            
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
    def InitializeDPArray(self, length: int):
        r"""
    
        Initialize the DP array and set all elements to 1.
        
        Args:
            length (int): The length of the DP array, which is usually equal to the length of the sequence.
            
        Returns:
            str: The initialized DP array.
            
        Example Output:
            "[1, 1, 1, 1, 1]"
        """
        self.dp = [1] * length
        return json.dumps(self.dp)
    
    def UpdateDPValue(self, i: int, j: int, sequence_i: int, sequence_j: int):
        r"""
    
        Update the value at index i in the DP array. If sequence_i >= sequence_j, update dp[i] to the maximum value between dp[i] and dp[j] + 1.
        
        Args:
            i (int): The index of the DP array to be updated currently.
            j (int): The index of the sequence element used for comparison.
            sequence_i (int): The element value at index i in the sequence.
            sequence_j (int): The element value at index j in the sequence.
            
        Returns:
            str: The updated value of dp[i].
            
        Example Output:
            "2"
        """
        if sequence_i >= sequence_j:
            self.dp[i] = max(self.dp[i], self.dp[j] + 1)
        return str(self.dp[i])
    
    def FindMaxDPValue(self):
        r"""
    
        Find the maximum value in the DP array, which is the length of the longest non-decreasing subsequence.
        
        Args:
            None
            
        Returns:
            str: The maximum value in the DP array.
            
        Example Output:
            "4"
        """
        if not self.dp:
            return "0"
        return str(max(self.dp))
    
    def Observe(self):
        r"""
    
        Obtain the sequence information in the current environment.
        
        Args:
            None
            
        Returns:
            str: The sequence in the current environment.
            
        Example Output:
            "[1, 3, 2, 4, 5]"
        """
        return json.dumps(self.sequence)
    
    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
        
        Args:
            answer (int): The length of the longest non-decreasing subsequence submitted by the user.
            
        Returns:
            str: Result information, including correctness and reward information.
            
        Example Output:
            "Your answer: 4, Reference answer: 4, Result: Correct, reward=1"
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
        sequence_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        sequence = ast.literal_eval(sequence_str)
        length = len(sequence)
        
        self.step(json.dumps({'name': 'InitializeDPArray', 'parameters': {'length': length}}))
        
        for i in range(length):
            for j in range(i):
                self.step(json.dumps({
                    'name': 'UpdateDPValue',
                    'parameters': {
                        'i': i,
                        'j': j,
                        'sequence_i': sequence[i],
                        'sequence_j': sequence[j]
                    }
                }))
        
        max_length = int(self.step(json.dumps({'name': 'FindMaxDPValue', 'parameters': {}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_sequence1 = [9, 8, 7, 6, 3, 0]
    env1 = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"sequence\": {test_sequence1}}}"
    )
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2
    print("\nTest Case 2:")
    test_sequence2 = [1, 3, 2, 4, 5]
    env2 = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"sequence\": {test_sequence2}}}"
    )
    print(env2.solve())
    print("step count:", env2.step_count)
    
    # test case 3 (random sequence)
    print("\nTest Case 3 (Random):")
    test_sequence3 = [random.randint(-10, 10) for _ in range(random.randint(1, 10))]
    env3 = LongestNonDecreasingSubsequenceEnv.from_env_str(
        f"LongestNonDecreasingSubsequenceEnv@{{\"sequence\": {test_sequence3}}}"
    )
    print(f"Sequence: {test_sequence3}")
    print(env3.solve())
    print("step count:", env3.step_count)