import gymnasium
import ast
import json
import random

class PartitionCheckerEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_TOTAL_WEIGHT = 0
        self.CHECK_IF_TOTAL_EVEN = 1
        self.CALCULATE_TARGET = 2
        self.CHECK_TARGET_LIMIT = 3
        self.INITIALIZE_DP_ARRAY = 4
        self.UPDATE_DP_ARRAY = 5
        self.CHECK_PARTITION_POSSIBLE = 6
        self.OBSERVE = 7
        self.DONE = 8

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateTotalWeight": self.CALCULATE_TOTAL_WEIGHT,
            "CheckIfTotalEven": self.CHECK_IF_TOTAL_EVEN,
            "CalculateTarget": self.CALCULATE_TARGET,
            "CheckTargetLimit": self.CHECK_TARGET_LIMIT,
            "InitializeDPArray": self.INITIALIZE_DP_ARRAY,
            "UpdateDPArray": self.UPDATE_DP_ARRAY,
            "CheckPartitionPossible": self.CHECK_PARTITION_POSSIBLE,
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
        prefix = "PartitionCheckerEnv@"
        if not env_str.startswith(prefix):
            return None
        return PartitionCheckerEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.weights = options.get("weights", [])
        self.limit = options.get("limit", 0)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        total_weight = sum(self.weights)
        
        if total_weight % 2 != 0:
            return False
        
        target = total_weight // 2
        
        if target > self.limit:
            return False
        
        dp = [False] * (target + 1)
        dp[0] = True
        
        for weight in self.weights:
            for j in range(target, weight - 1, -1):
                dp[j] = dp[j] or dp[j - weight]
        
        return dp[target]

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
            
            if action_code == self.CALCULATE_TOTAL_WEIGHT:
                msg = self.CalculateTotalWeight()
            
            elif action_code == self.CHECK_IF_TOTAL_EVEN:
                if "total_weight" in params:
                    total_weight = params["total_weight"]
                    msg = self.CheckIfTotalEven(total_weight)
                else:
                    msg = "Error: 'total_weight' parameter is missing for CHECK_IF_TOTAL_EVEN action."
            
            elif action_code == self.CALCULATE_TARGET:
                if "total_weight" in params:
                    total_weight = params["total_weight"]
                    msg = self.CalculateTarget(total_weight)
                else:
                    msg = "Error: 'total_weight' parameter is missing for CALCULATE_TARGET action."
            
            elif action_code == self.CHECK_TARGET_LIMIT:
                if "target" in params:
                    target = params["target"]
                    msg = self.CheckTargetLimit(target)
                else:
                    msg = "Error: 'target' parameter is missing for CHECK_TARGET_LIMIT action."
            
            elif action_code == self.INITIALIZE_DP_ARRAY:
                if "target" in params:
                    target = params["target"]
                    msg = self.InitializeDPArray(target)
                else:
                    msg = "Error: 'target' parameter is missing for INITIALIZE_DP_ARRAY action."
            
            elif action_code == self.UPDATE_DP_ARRAY:
                if "dp_array" in params and "weight" in params and "target" in params:
                    dp_array = params["dp_array"]
                    weight = params["weight"]
                    target = params["target"]
                    msg = self.UpdateDPArray(dp_array, weight, target)
                else:
                    msg = "Error: 'dp_array', 'weight' or 'target' parameter is missing for UPDATE_DP_ARRAY action."
            
            elif action_code == self.CHECK_PARTITION_POSSIBLE:
                if "dp_array" in params and "target" in params:
                    dp_array = params["dp_array"]
                    target = params["target"]
                    msg = self.CheckPartitionPossible(dp_array, target)
                else:
                    msg = "Error: 'dp_array' or 'target' parameter is missing for CHECK_PARTITION_POSSIBLE action."
            
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
    def CalculateTotalWeight(self):
        r"""
    
        Calculate the sum of all elements in the weights list.
    
        Args:
            None
    
        Returns:
            str: The total sum of the weight list.
    
        Example Output:
            "10"
        """
        total = sum(self.weights)
        return str(total)

    def CheckIfTotalEven(self, total_weight: int):
        r"""
    
        Check if the total weight is an even number.
    
        Args:
            total_weight (int): The total weight.
    
        Returns:
            str: A boolean string indicating whether the total weight is even.
    
        Example Output:
            "True"
        """
        return str(total_weight % 2 == 0)

    def CalculateTarget(self, total_weight: int):
        r"""
    
        Calculate the target value, which is half of the total weight.
    
        Args:
            total_weight (int): The total weight.
    
        Returns:
            str: The target value.
    
        Example Output:
            "5"
        """
        return str(total_weight // 2)

    def CheckTargetLimit(self, target: int):
        r"""
    
        Check if the target value exceeds the weight limit.
    
        Args:
            target (int): The target value.
    
        Returns:
            str: A boolean string indicating whether the target value exceeds the weight limit.
    
        Example Output:
            "False"
        """
        return str(target > self.limit)

    def InitializeDPArray(self, target: int):
        r"""
    
        Initialize the dynamic programming array, with the first element being True and the rest being False.
    
        Args:
            target (int): The target value.
    
        Returns:
            str: The initialized dynamic programming array.
    
        Example Output:
            "[True, False, False, False, False, False]"
        """
        dp = [False] * (target + 1)
        dp[0] = True
        return str(dp)

    def UpdateDPArray(self, dp_array: list, weight: int, target: int):
        r"""
    
        Update the dynamic programming array based on the current weight.
    
        Args:
            dp_array (list[bool]): The current dynamic programming array.
            weight (int): The current weight to be processed.
            target (int): The target value.
    
        Returns:
            str: The updated dynamic programming array.
    
        Example Output:
            "[True, False, True, False, True, False]"
        """
        bool_dp = [x == 'True' for x in dp_array] if isinstance(dp_array[0], str) else dp_array
        
        for j in range(target, weight - 1, -1):
            bool_dp[j] = bool_dp[j] or bool_dp[j - weight]
        
        return str(bool_dp)

    def CheckPartitionPossible(self, dp_array: list, target: int):
        r"""
    
        Check if partitioning is possible, i.e., check if dp_array[target] is True.
    
        Args:
            dp_array (list[bool]): The dynamic programming array.
            target (int): The target value.
    
        Returns:
            str: A boolean string indicating whether partitioning is possible.
    
        Example Output:
            "True"
        """
        bool_dp = [x == 'True' for x in dp_array] if isinstance(dp_array[0], str) else dp_array
        return str(bool_dp[target])

    def Observe(self):
        r"""
    
        Return the weight list and weight limit in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current weight list and weight limit.
    
        Example Output:
            "weights: [1, 2, 3, 4], limit: 5"
        """
        return f"weights: {self.weights}, limit: {self.limit}"

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return the result information.
    
        Args:
            answer (bool): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: True, Reference answer: True, Result: Correct, reward=1"
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
        limit = int(observe_info.split('limit: ')[1])
        
        total_weight_str = self.step(json.dumps({'name': 'CalculateTotalWeight', 'parameters': {}}))[1]
        total_weight = int(total_weight_str)
        
        if total_weight > 2 * limit:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
        
        is_even_str = self.step(json.dumps({'name': 'CheckIfTotalEven', 'parameters': {'total_weight': total_weight}}))[1]
        is_even = is_even_str == "True"
        if not is_even:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
        
        target_str = self.step(json.dumps({'name': 'CalculateTarget', 'parameters': {'total_weight': total_weight}}))[1]
        target = int(target_str)
        
        target_exceed_str = self.step(json.dumps({'name': 'CheckTargetLimit', 'parameters': {'target': target}}))[1]
        target_exceed = target_exceed_str == "True"
        if target_exceed:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
        
        dp_init_str = self.step(json.dumps({'name': 'InitializeDPArray', 'parameters': {'target': target}}))[1]
        dp_array = ast.literal_eval(dp_init_str)
        
        weights_str = observe_info.split('weights: ')[1].split(', limit:')[0]
        weights = ast.literal_eval(weights_str)
        
        for weight in weights:
            dp_array = ast.literal_eval(self.step(json.dumps({
                'name': 'UpdateDPArray',
                'parameters': {'dp_array': dp_array, 'weight': weight, 'target': target}
            }))[1])
        
        partition_possible_str = self.step(json.dumps({
            'name': 'CheckPartitionPossible',
            'parameters': {'dp_array': dp_array, 'target': target}
        }))[1]
        partition_possible = partition_possible_str == "True"
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': partition_possible}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_weights1 = [1, 2, 3, 4]
    test_limit1 = 5
    env1 = PartitionCheckerEnv.from_env_str(f"PartitionCheckerEnv@{{\"weights\": {test_weights1}, \"limit\": {test_limit1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_weights2 = [3, 3, 3, 3]
    test_limit2 = 5
    env2 = PartitionCheckerEnv.from_env_str(f"PartitionCheckerEnv@{{\"weights\": {test_weights2}, \"limit\": {test_limit2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)
    
    # test case 3 (random)
    print("\nTest Case 3 (Random):")
    random_weights = [random.randint(1, 10) for _ in range(random.randint(3, 8))]
    random_limit = random.randint(5, 30)
    env3 = PartitionCheckerEnv.from_env_str(f"PartitionCheckerEnv@{{\"weights\": {random_weights}, \"limit\": {random_limit}}}")
    print(env3.solve())
    print("step count:", env3.step_count)