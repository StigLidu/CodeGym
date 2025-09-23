import gymnasium
import ast
import json
import random

class SmallestSubsetEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.SORT_POINTS = 0
        self.SELECT_POINT = 1
        self.FIND_FARTHEST_WITHIN_DISTANCE = 2
        self.SKIP_POINTS_WITHIN_DISTANCE = 3
        self.OBSERVE = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "SortPoints": self.SORT_POINTS,
            "SelectPoint": self.SELECT_POINT,
            "FindFarthestWithinDistance": self.FIND_FARTHEST_WITHIN_DISTANCE,
            "SkipPointsWithinDistance": self.SKIP_POINTS_WITHIN_DISTANCE,
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
        prefix = "SmallestSubsetEnv@"
        if not env_str.startswith(prefix):
            return None
        return SmallestSubsetEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.d = options.get("d", 0)
        self.points = options.get("points", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        points = sorted(self.points)
        subset_size = 0
        i = 0

        while i < self.n:
            subset_size += 1
            target = points[i] + self.d
            while i < self.n and points[i] <= target:
                i += 1
            if i < self.n:
                target = points[i - 1] + self.d
                while i < self.n and points[i] <= target:
                    i += 1

        return subset_size

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
            
            if action_code == self.SORT_POINTS:
                if "points" in params:
                    points = params["points"]
                    msg = self.SortPoints(points)
                else:
                    msg = "Error: 'points' parameter is missing for SORT_POINTS action."
            
            elif action_code == self.SELECT_POINT:
                if "points" in params and "index" in params:
                    points = params["points"]
                    index = params["index"]
                    msg = self.SelectPoint(points, index)
                else:
                    msg = "Error: 'points' or 'index' parameter is missing for SELECT_POINT action."
                    
            elif action_code == self.FIND_FARTHEST_WITHIN_DISTANCE:
                if "points" in params and "start_index" in params and "distance" in params:
                    points = params["points"]
                    start_index = params["start_index"]
                    distance = params["distance"]
                    msg = self.FindFarthestWithinDistance(points, start_index, distance)
                else:
                    msg = "Error: 'points', 'start_index' or 'distance' parameter is missing for FIND_FARTHEST_WITHIN_DISTANCE action."
                    
            elif action_code == self.SKIP_POINTS_WITHIN_DISTANCE:
                if "points" in params and "start_index" in params and "target" in params:
                    points = params["points"]
                    start_index = params["start_index"]
                    target = params["target"]
                    msg = self.SkipPointsWithinDistance(points, start_index, target)
                else:
                    msg = "Error: 'points', 'start_index' or 'target' parameter is missing for SKIP_POINTS_WITHIN_DISTANCE action."
                    
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
    def SortPoints(self, points: list):
        r"""
    
        Sorts the input list of points.
    
        Args:
            points (list[int]): The list of points to be sorted.
    
        Returns:
            str: The sorted list of points, converted to a string using json.dumps.
    
        Example Output:
            "[1, 2, 4, 5, 6]"
        """
        sorted_points = sorted(points)
        return json.dumps(sorted_points)

    def SelectPoint(self, points: list, index: int):
        r"""
    
        Selects the point at the specified index from the list of points.
    
        Args:
            points (list[int]): The list of points.
            index (int): The index of the point to be selected.
    
        Returns:
            str: The value of the selected point.
    
        Example Output:
            "1"
        """
        if 0 <= index < len(points):
            return str(points[index])
        return "Error: Index out of range"

    def FindFarthestWithinDistance(self, points: list, start_index: int, distance: int):
        r"""
    
        Starting from the initial index, finds the index of the farthest point that is no more than the specified distance away from the starting point.
    
        Args:
            points (list[int]): The list of points.
            start_index (int): The initial index.
            distance (int): The maximum allowed distance.
    
        Returns:
            str: The index of the found farthest point, converted to a string using json.dumps.
    
        Example Output:
            "1"
        """
        if start_index >= len(points):
            return json.dumps(len(points))
            
        target = points[start_index] + distance
        i = start_index
        while i < len(points) and points[i] <= target:
            i += 1
        return json.dumps(i - 1)

    def SkipPointsWithinDistance(self, points: list, start_index: int, target: int):
        r"""
    
        Starting from the initial index, skips all points that do not exceed the target value and returns the index of the first point that exceeds the target value.
    
        Args:
            points (list[int]): The list of points.
            start_index (int): The initial index.
            target (int): The target value.
    
        Returns:
            str: The index of the first point exceeding the target value, converted to a string using json.dumps.
    
        Example Output:
            "2"
        """
        i = start_index
        while i < len(points) and points[i] <= target:
            i += 1
        return json.dumps(i)

    def Observe(self):
        r"""
    
        Obtains the status information of the current environment, including the number of points n, the distance d, and the list of points.
    
        Args:
            None
    
        Returns:
            str: The status information of the current environment, converted to a string using json.dumps.
    
        Example Output:
            "{\"n\": 5, \"d\": 2, \"points\": [1, 2, 4, 5, 6]}"
        """
        state = {
            "n": self.n,
            "d": self.d,
            "points": self.points
        }
        return json.dumps(state)

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (int): The minimum subset size submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        obs_dict = json.loads(obs)
        points = obs_dict['points']
        d = obs_dict['d']
        
        sorted_points_str = self.step(json.dumps({'name': 'SortPoints', 'parameters': {'points': points}}))[1]
        sorted_points = json.loads(sorted_points_str)
        
        count = 0
        i = 0
        n = len(sorted_points)
        
        while i < n:
            start_point_str = self.step(json.dumps({
                'name': 'SelectPoint', 
                'parameters': {'points': sorted_points, 'index': i}
            }))[1]
            start_point = int(start_point_str)
            
            farthest_idx_str = self.step(json.dumps({
                'name': 'FindFarthestWithinDistance', 
                'parameters': {
                    'points': sorted_points, 
                    'start_index': i, 
                    'distance': d
                }
            }))[1]
            farthest_idx = json.loads(farthest_idx_str)
            
            count += 1
            
            selected_point_str = self.step(json.dumps({
                'name': 'SelectPoint', 
                'parameters': {'points': sorted_points, 'index': farthest_idx}
            }))[1]
            selected_point = int(selected_point_str)
            
            target = selected_point + d
            next_i_str = self.step(json.dumps({
                'name': 'SkipPointsWithinDistance', 
                'parameters': {
                    'points': sorted_points, 
                    'start_index': farthest_idx, 
                    'target': target
                }
            }))[1]
            i = json.loads(next_i_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (example from problem statement)
    print("Test Case 1:")
    test_env_str1 = "SmallestSubsetEnv@{\"n\": 5, \"d\": 2, \"points\": [1, 2, 4, 5, 6]}"
    env1 = SmallestSubsetEnv.from_env_str(test_env_str1)
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2 (random test case)
    print("\nTest Case 2:")
    n2 = random.randint(1, 10)
    d2 = random.randint(1, 5)
    points2 = [random.randint(1, 20) for _ in range(n2)]
    test_env_str2 = f"SmallestSubsetEnv@{{\"n\": {n2}, \"d\": {d2}, \"points\": {points2}}}"
    env2 = SmallestSubsetEnv.from_env_str(test_env_str2)
    print(env2.solve())
    print("step count:", env2.step_count)