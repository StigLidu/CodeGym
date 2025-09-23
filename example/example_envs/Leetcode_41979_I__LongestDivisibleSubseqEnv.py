import gymnasium
import ast
import json
from copy import copy

class LongestDivisibleSubseqEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.INITIALIZE_DP = 1
        self.PROCESS_NUMBER = 2
        self.GET_MAX_LENGTH = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "InitializeDP": self.INITIALIZE_DP,
            "ProcessNumber": self.PROCESS_NUMBER,
            "GetMaxLength": self.GET_MAX_LENGTH,
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
        prefix = "LongestDivisibleSubseqEnv@"
        if not env_str.startswith(prefix):
            return None
        return LongestDivisibleSubseqEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.sequence = options.get("sequence", [])
        self.k = options.get("k", 1)
        self.dp = None
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if self.k == 0:
            return 0
            
        dp = {0: 0}
        for num in self.sequence:
            current_dp = dp.copy()
            for modulus, length in current_dp.items():
                new_modulus = (modulus + num) % self.k
                new_length = length + 1
                if new_modulus not in dp or new_length > dp[new_modulus]:
                    dp[new_modulus] = new_length
        return dp.get(0, 0)

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
            
            elif action_code == self.INITIALIZE_DP:
                msg = self.InitializeDP()
                
            elif action_code == self.PROCESS_NUMBER:
                if "number" in params:
                    number = params["number"]
                    msg = self.ProcessNumber(number)
                else:
                    msg = "Error: 'number' parameter is missing for PROCESS_NUMBER action."
                    
            elif action_code == self.GET_MAX_LENGTH:
                msg = self.GetMaxLength()
                
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
    
        Obtain the sequence and k value in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the sequence and k value.
    
        Example Output:
            "{\"sequence\": [3, 1, 2, 6], \"k\": 3}"
        """
        return json.dumps({"sequence": self.sequence, "k": self.k})

    def InitializeDP(self):
        r"""
    
        Initialize the dynamic programming dictionary to track the length of the longest subsequence corresponding to each modulo value.
    
        Args:
            None
    
        Returns:
            str: A JSON string of the initialized DP dictionary.
    
        Example Output:
            "{\"0\": 0}"
        """
        self.dp = {0: 0}
        return json.dumps(self.dp)

    def ProcessNumber(self, number: int):
        r"""
    
        Process a number in the sequence and update the length of the longest subsequence corresponding to each modulo value in the DP dictionary.
    
        Args:
            number (int): The number to be processed.
    
        Returns:
            str: A JSON string of the processed DP dictionary.
    
        Example Output:
            "{\"0\": 0, \"1\": 1}"
        """
        if self.dp is None:
            return "Error: DP dictionary not initialized. Call InitializeDP first."
            
        current_dp = copy(self.dp)
        for modulus, length in current_dp.items():
            new_modulus = (modulus + number) % self.k
            new_length = length + 1
            if new_modulus not in self.dp or new_length > self.dp[new_modulus]:
                self.dp[new_modulus] = new_length
                
        return json.dumps(self.dp)

    def GetMaxLength(self):
        r"""
    
        Obtain the length of the longest subsequence whose sum is divisible by k.
    
        Args:
            None
    
        Returns:
            str: The length of the longest subsequence.
    
        Example Output:
            "4"
        """
        if self.dp is None:
            return "Error: DP dictionary not initialized. Call InitializeDP first."
            
        return str(self.dp.get(0, 0))

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        observe_data = json.loads(observe_result)
        sequence = observe_data['sequence']
        k = observe_data['k']
        
        self.step(json.dumps({'name': 'InitializeDP', 'parameters': {}}))
        
        for num in sequence:
            self.step(json.dumps({'name': 'ProcessNumber', 'parameters': {'number': num}}))
        
        max_length = self.step(json.dumps({'name': 'GetMaxLength', 'parameters': {}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': int(max_length)}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_sequence = [3, 1, 2, 6]
    test_k = 3
    env = LongestDivisibleSubseqEnv.from_env_str(
        f"LongestDivisibleSubseqEnv@{{\"sequence\": {test_sequence}, \"k\": {test_k}}}"
    )
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_sequence = [5, 0, 2, 3, 1]
    test_k = 5
    env = LongestDivisibleSubseqEnv.from_env_str(
        f"LongestDivisibleSubseqEnv@{{\"sequence\": {test_sequence}, \"k\": {test_k}}}"
    )
    print(env.solve())
    print("step count:", env.step_count)