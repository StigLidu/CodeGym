import gymnasium
import ast
import json

class PlantCollectionEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_DIRECTION = 1
        self.CALCULATE_MAX_FROM_POSITION = 2
        self.FIND_OVERALL_MAXIMUM = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckDirection": self.CHECK_DIRECTION,
            "CalculateMaxFromPosition": self.CALCULATE_MAX_FROM_POSITION,
            "FindOverallMaximum": self.FIND_OVERALL_MAXIMUM,
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
        prefix = "PlantCollectionEnv@"
        if not env_str.startswith(prefix):
            return None
        return PlantCollectionEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.heights = options.get("heights", [])
        self.d = options.get("d", 0)
        self.n = len(self.heights)
        self.dp = [0] * self.n  # Memoization array
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def dfs(i):
            if self.dp[i] != 0:
                return self.dp[i]
            
            max_collect = 1
            
            for direction in [-1, 1]:  # -1 for left, 1 for right
                for j in range(1, self.d+1):
                    ni = i + direction * j
                    if 0 <= ni < self.n and self.heights[ni] > self.heights[i]:
                        max_collect = max(max_collect, 1 + dfs(ni))
                    elif 0 <= ni < self.n and self.heights[ni] <= self.heights[i]:
                        break
            
            self.dp[i] = max_collect
            return max_collect
        
        max_plants = 0
        for i in range(self.n):
            max_plants = max(max_plants, dfs(i))
        
        return max_plants

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
            
            elif action_code == self.CHECK_DIRECTION:
                if "position" in params and "direction" in params and "distance" in params:
                    position = params["position"]
                    direction = params["direction"]
                    distance = params["distance"]
                    msg = self.CheckDirection(position, direction, distance)
                else:
                    msg = "Error: 'position', 'direction' or 'distance' parameter is missing for CHECK_DIRECTION action."
            
            elif action_code == self.CALCULATE_MAX_FROM_POSITION:
                if "position" in params:
                    position = params["position"]
                    msg = self.CalculateMaxFromPosition(position)
                else:
                    msg = "Error: 'position' parameter is missing for CALCULATE_MAX_FROM_POSITION action."
            
            elif action_code == self.FIND_OVERALL_MAXIMUM:
                if "values" in params:
                    values = params["values"]
                    msg = self.FindOverallMaximum(values)
                else:
                    msg = "Error: 'values' parameter is missing for FIND_OVERALL_MAXIMUM action."
            
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
    
        Obtain the array of plant heights and the maximum jumping distance in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the plant heights and the maximum jumping distance.
    
        Example Output:
            "{\"heights\": [3, 4, 2, 1, 5], \"d\": 2}"
        """
        observation = {
            "heights": self.heights,
            "d": self.d
        }
        return json.dumps(observation)

    def CheckDirection(self, position: int, direction: int, distance: int):
        r"""
    
        Check if there is a taller plant after jumping a specified distance in a specific direction from the designated position.
    
        Args:
            position (int): The index of the current plant position.
            direction (int): The jumping direction, where 1 indicates right and -1 indicates left.
            distance (int): The jumping distance, which must be an integer between 1 and d.
    
        Returns:
            str: A JSON string indicating whether the target position is valid and whether the plant there is taller.
    
        Example Output:
            "{\"valid\": true, \"higher\": true, \"position\": 1}"
        """
        if position < 0 or position >= self.n:
            return json.dumps({"valid": False, "higher": False, "position": -1})
            
        target_pos = position + direction * distance
        
        result = {
            "valid": 0 <= target_pos < self.n,
            "higher": False,
            "position": target_pos
        }
        
        if result["valid"]:
            result["higher"] = self.heights[target_pos] > self.heights[position]
            
        return json.dumps(result)

    def CalculateMaxFromPosition(self, position: int):
        r"""
    
        Calculate the maximum number of plants that can be collected starting from the specified position.
    
        Args:
            position (int): The index of the starting plant position.
    
        Returns:
            str: The maximum number of plants that can be collected starting from this position.
    
        Example Output:
            "3"
        """
        if self.dp[position] != 0:
            return str(self.dp[position])
            
        max_collect = 1
        
        # Check both directions
        for direction in [-1, 1]:
            for j in range(1, self.d + 1):
                target_pos = position + direction * j
                if 0 <= target_pos < self.n and self.heights[target_pos] > self.heights[position]:
                    # Recursively get the max from target position
                    action = json.dumps({
                        "name": "CalculateMaxFromPosition",
                        "parameters": {"position": target_pos}
                    })
                    sub_result = int(self.step(action)[1])
                    max_collect = max(max_collect, 1 + sub_result)
                elif 0 <= target_pos < self.n and self.heights[target_pos] <= self.heights[position]:
                    break
                    
        self.dp[position] = max_collect
        return str(max_collect)

    def FindOverallMaximum(self, values: list):
        r"""
    
        Find the maximum value in the list of numbers.
    
        Args:
            values (list[int]): A list of integers.
    
        Returns:
            str: The maximum value in the list.
    
        Example Output:
            "5"
        """
        return str(max(values))

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
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

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        obs = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        obs_dict = json.loads(obs)
        heights = obs_dict['heights']
        n = len(heights)
        
        max_from_positions = []
        for pos in range(n):
            count = int(self.step(json.dumps({'name': 'CalculateMaxFromPosition', 'parameters': {'position': pos}}))[1])
            max_from_positions.append(count)
        
        overall_max = int(self.step(json.dumps({'name': 'FindOverallMaximum', 'parameters': {'values': max_from_positions}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': overall_max}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_heights1 = [3, 4, 2, 1, 5]
    test_d1 = 2
    env1 = PlantCollectionEnv.from_env_str(f"PlantCollectionEnv@{{\"heights\": {test_heights1}, \"d\": {test_d1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_heights2 = [6, 4, 14, 6, 8, 13, 9, 7, 10, 6, 12]
    test_d2 = 2
    env2 = PlantCollectionEnv.from_env_str(f"PlantCollectionEnv@{{\"heights\": {test_heights2}, \"d\": {test_d2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)