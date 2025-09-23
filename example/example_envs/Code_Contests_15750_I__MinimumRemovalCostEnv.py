import gymnasium
import ast
import json

class MinimumRemovalCostEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_SUBARRAY_SUM = 0
        self.UPDATE_DP_VALUE = 1
        self.GET_DP_VALUE = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateSubarraySum": self.CALCULATE_SUBARRAY_SUM,
            "UpdateDpValue": self.UPDATE_DP_VALUE,
            "GetDpValue": self.GET_DP_VALUE,
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
        prefix = "MinimumRemovalCostEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinimumRemovalCostEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.k = options.get("k", 0)
        self.arr = options.get("arr", [])
        self.dp = [float('inf')] * (self.n + 1)
        self.dp[0] = 0  # No cost to remove zero elements
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        dp = [float('inf')] * (self.n + 1)
        dp[0] = 0  # No cost to remove zero elements

        for i in range(1, self.n + 1):
            for j in range(1, self.k + 1):
                if i - j >= 0:
                    dp[i] = min(dp[i], dp[i - j] + sum(self.arr[i - j: i]))

        return dp[self.n]

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
            
            if action_code == self.CALCULATE_SUBARRAY_SUM:
                if "start_index" in params and "end_index" in params:
                    start_index = params["start_index"]
                    end_index = params["end_index"]
                    msg = self.CalculateSubarraySum(start_index, end_index)
                else:
                    msg = "Error: 'start_index' or 'end_index' parameter is missing for CALCULATE_SUBARRAY_SUM action."
            
            elif action_code == self.UPDATE_DP_VALUE:
                if "index" in params and "value" in params:
                    index = params["index"]
                    value = params["value"]
                    msg = self.UpdateDpValue(index, value)
                else:
                    msg = "Error: 'index' or 'value' parameter is missing for UPDATE_DP_VALUE action."
                    
            elif action_code == self.GET_DP_VALUE:
                if "index" in params:
                    index = params["index"]
                    msg = self.GetDpValue(index)
                else:
                    msg = "Error: 'index' parameter is missing for GET_DP_VALUE action."
                    
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
    def CalculateSubarraySum(self, start_index: int, end_index: int):
        r"""
    
        Calculate the sum of the subarray from start_index to end_index (exclusive).
    
        Args:
            start_index (int): The starting index of the subarray (inclusive).
            end_index (int): The ending index of the subarray (exclusive).
    
        Returns:
            str: The sum of the subarray.
    
        Example Output:
            "8"
        """
        subarray_sum = sum(self.arr[start_index:end_index])
        return str(subarray_sum)

    def UpdateDpValue(self, index: int, value: int):
        r"""
    
        Update the value at a specific index in the dp array.
    
        Args:
            index (int): The index of the dp array.
            value (int): The new value to be set.
    
        Returns:
            str: The updated dp value.
    
        Example Output:
            "4"
        """
        self.dp[index] = value
        return str(self.dp[index])

    def GetDpValue(self, index: int):
        r"""
    
        Get the value at a specific index in the dp array.
    
        Args:
            index (int): The index of the dp array.
    
        Returns:
            str: The value at that index in the dp array.
    
        Example Output:
            "7"
        """
        return str(self.dp[index])

    def Observe(self):
        r"""
    
        Return the observation information of the current environment, including the array length n, the maximum subarray length k, and the array content.
    
        Args:
            None
    
        Returns:
            str: The observation information of the environment.
    
        Example Output:
            "n=5, k=2, arr=[1, 3, 5, 2, 4]"
        """
        return f"n={self.n}, k={self.k}, arr={self.arr}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 15, Reference answer: 15, Result: Correct, reward=1"
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
        observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = int(observe_info.split(',')[0].split('=')[1].strip())
        k = int(observe_info.split(',')[1].split('=')[1].strip())
        
        self.step(json.dumps({'name': 'UpdateDpValue', 'parameters': {'index': 0, 'value': 0}}))
        
        for i in range(1, n + 1):
            min_cost = float('inf')
            start_j = max(0, i - k)
            for j in range(start_j, i):
                sum_str = self.step(json.dumps({
                    'name': 'CalculateSubarraySum',
                    'parameters': {'start_index': j, 'end_index': i}
                }))[1]
                sub_sum = int(sum_str)
                dp_j_str = self.step(json.dumps({
                    'name': 'GetDpValue',
                    'parameters': {'index': j}
                }))[1]
                dp_j = int(dp_j_str)
                current_cost = dp_j + sub_sum
                if current_cost < min_cost:
                    min_cost = current_cost
            self.step(json.dumps({
                'name': 'UpdateDpValue',
                'parameters': {'index': i, 'value': min_cost}
            }))
        
        answer_str = self.step(json.dumps({
            'name': 'GetDpValue',
            'parameters': {'index': n}
        }))[1]
        answer = int(answer_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_env_str1 = "MinimumRemovalCostEnv@{\"n\": 5, \"k\": 2, \"arr\": [1, 3, 5, 2, 4]}"
    env1 = MinimumRemovalCostEnv.from_env_str(test_env_str1)
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_env_str2 = "MinimumRemovalCostEnv@{\"n\": 6, \"k\": 3, \"arr\": [2, -1, 2, 3, 4, -5]}"
    env2 = MinimumRemovalCostEnv.from_env_str(test_env_str2)
    print(env2.solve())
    print("step count:", env2.step_count)