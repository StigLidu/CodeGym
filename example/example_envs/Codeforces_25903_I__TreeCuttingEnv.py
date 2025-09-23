import gymnasium
import ast
import json
from bisect import bisect_right

class TreeCuttingEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.MODIFY_HEIGHT = 1
        self.CALCULATE_LIS = 2
        self.CALCULATE_CUTS = 3
        self.FIND_MIN_CUTS = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "ModifyHeight": self.MODIFY_HEIGHT,
            "CalculateLIS": self.CALCULATE_LIS,
            "CalculateCuts": self.CALCULATE_CUTS,
            "FindMinCuts": self.FIND_MIN_CUTS,
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
        prefix = "TreeCuttingEnv@"
        if not env_str.startswith(prefix):
            return None
        return TreeCuttingEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.k = options.get("k", 0)
        self.heights = options.get("heights", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def longest_non_decreasing_subsequence(seq):
            lis = []
            for h in seq:
                pos = bisect_right(lis, h)
                if pos < len(lis):
                    lis[pos] = h
                else:
                    lis.append(h)
            return len(lis)
        
        best_result = float('inf')
        
        for i in range(self.n + 1):
            if i < self.n:
                modified_heights = self.heights[:i] + [self.heights[i] + self.k] + self.heights[i+1:]
            else:
                modified_heights = self.heights
            
            lis_length = longest_non_decreasing_subsequence(modified_heights)
            cuts_required = self.n - lis_length
            
            best_result = min(best_result, cuts_required)
        
        return best_result

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
            
            elif action_code == self.MODIFY_HEIGHT:
                if "index" in params and "original_heights" in params and "k" in params:
                    index = params["index"]
                    original_heights = params["original_heights"]
                    k_val = params["k"]
                    msg = self.ModifyHeight(index, original_heights, k_val)
                else:
                    msg = "Error: 'index', 'original_heights' or 'k' parameter is missing for MODIFY_HEIGHT action."
            
            elif action_code == self.CALCULATE_LIS:
                if "sequence" in params:
                    sequence = params["sequence"]
                    msg = self.CalculateLIS(sequence)
                else:
                    msg = "Error: 'sequence' parameter is missing for CALCULATE_LIS action."
            
            elif action_code == self.CALCULATE_CUTS:
                if "n" in params and "lis_length" in params:
                    n_val = params["n"]
                    lis_length = params["lis_length"]
                    msg = self.CalculateCuts(n_val, lis_length)
                else:
                    msg = "Error: 'n' or 'lis_length' parameter is missing for CALCULATE_CUTS action."
            
            elif action_code == self.FIND_MIN_CUTS:
                if "cuts_list" in params:
                    cuts_list = params["cuts_list"]
                    msg = self.FindMinCuts(cuts_list)
                else:
                    msg = "Error: 'cuts_list' parameter is missing for FIND_MIN_CUTS action."
            
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
    
        Obtain the number of trees, the height increment, and the list of tree heights in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the number of trees, the height increment, and the list of tree heights.
    
        Example Output:
            "{\"n\": 5, \"k\": 3, \"heights\": [3, 1, 2, 6, 4]}"
        """
        observation = {
            "n": self.n,
            "k": self.k,
            "heights": self.heights
        }
        return json.dumps(observation)

    def ModifyHeight(self, index: int, original_heights: list, k: int):
        r"""
    
        Increase the height of the tree at the specified index by k units and return the modified height list.
    
        Args:
            index (int): The index of the tree whose height is to be modified; if it is n, it means no tree is modified.
            original_heights (list[int]): The original list of tree heights.
            k (int): The height increment.
    
        Returns:
            str: A JSON string of the modified list of tree heights.
    
        Example Output:
            "[3, 1, 5, 6, 4]"
        """
        n = len(original_heights)
        if index < n:
            modified_heights = original_heights[:index] + [original_heights[index] + k] + original_heights[index+1:]
        else:
            modified_heights = original_heights.copy()
        return json.dumps(modified_heights)

    def CalculateLIS(self, sequence: list):
        r"""
    
        Calculate the length of the longest non-decreasing subsequence of a given sequence.
    
        Args:
            sequence (list[int]): The sequence to be calculated.
    
        Returns:
            str: The length of the longest non-decreasing subsequence.
    
        Example Output:
            "4"
        """
        lis = []
        for h in sequence:
            pos = bisect_right(lis, h)
            if pos < len(lis):
                lis[pos] = h
            else:
                lis.append(h)
        return str(len(lis))

    def CalculateCuts(self, n: int, lis_length: int):
        r"""
    
        Calculate the number of trees that need to be cut down.
    
        Args:
            n (int): The total number of trees.
            lis_length (int): The length of the longest non-decreasing subsequence.
    
        Returns:
            str: The number of trees that need to be cut down.
    
        Example Output:
            "1"
        """
        cuts_required = n - lis_length
        return str(cuts_required)

    def FindMinCuts(self, cuts_list: list):
        r"""
    
        Find the minimum number of trees that need to be cut down.
    
        Args:
            cuts_list (list[int]): A list of the number of trees that need to be cut down under different schemes.
    
        Returns:
            str: The minimum number of trees that need to be cut down.
    
        Example Output:
            "1"
        """
        min_cuts = min(cuts_list)
        return str(min_cuts)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
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
        import json
        
        obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        obs_data = json.loads(obs)
        n = obs_data['n']
        k = obs_data['k']
        original_heights = obs_data['heights']
        
        cuts_list = []
        
        lis_length = int(self.step(json.dumps({
            'name': 'CalculateLIS',
            'parameters': {'sequence': original_heights}
        }))[1])
        cuts = int(self.step(json.dumps({
            'name': 'CalculateCuts',
            'parameters': {'n': n, 'lis_length': lis_length}
        }))[1])
        cuts_list.append(cuts)
        
        for index in range(n):
            modified_heights_str = self.step(json.dumps({
                'name': 'ModifyHeight',
                'parameters': {'index': index, 'original_heights': original_heights, 'k': k}
            }))[1]
            modified_heights = json.loads(modified_heights_str)
            
            lis_len_mod = int(self.step(json.dumps({
                'name': 'CalculateLIS',
                'parameters': {'sequence': modified_heights}
            }))[1])
            
            current_cuts = int(self.step(json.dumps({
                'name': 'CalculateCuts',
                'parameters': {'n': n, 'lis_length': lis_len_mod}
            }))[1])
            cuts_list.append(current_cuts)
        
        min_cuts = int(self.step(json.dumps({
            'name': 'FindMinCuts',
            'parameters': {'cuts_list': cuts_list}
        }))[1])
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': min_cuts}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - example from problem
    print("Test Case 1:")
    env1 = TreeCuttingEnv.from_env_str("TreeCuttingEnv@{\"n\": 5, \"k\": 3, \"heights\": [3, 1, 2, 6, 4]}")
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2 - random case
    print("\nTest Case 2:")
    env2 = TreeCuttingEnv.from_env_str("TreeCuttingEnv@{\"n\": 6, \"k\": 2, \"heights\": [5, 3, 4, 7, 1, 8]}")
    print(env2.solve())
    print("step count:", env2.step_count)