import gymnasium
import ast
import json

class UniqueVisibleIntervalsEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_INTERVAL_UNIQUENESS = 1
        self.COUNT_INTERVALS_STARTING_AT = 2
        self.SUM_COUNTS = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckIntervalUniqueness": self.CHECK_INTERVAL_UNIQUENESS,
            "CountIntervalsStartingAt": self.COUNT_INTERVALS_STARTING_AT,
            "SumCounts": self.SUM_COUNTS,
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
        prefix = "UniqueVisibleIntervalsEnv@"
        if not env_str.startswith(prefix):
            return None
        return UniqueVisibleIntervalsEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
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
        n = len(self.heights)
        count = 0
        
        for i in range(n):
            seen = set()
            for j in range(i, n):
                if self.heights[j] in seen:
                    break
                seen.add(self.heights[j])
                count += 1
        
        return count

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
                
            elif action_code == self.CHECK_INTERVAL_UNIQUENESS:
                if "start" in params and "end" in params:
                    start = params["start"]
                    end = params["end"]
                    msg = self.CheckIntervalUniqueness(start, end)
                else:
                    msg = "Error: 'start' or 'end' parameter is missing for CHECK_INTERVAL_UNIQUENESS action."
                    
            elif action_code == self.COUNT_INTERVALS_STARTING_AT:
                if "start" in params:
                    start = params["start"]
                    msg = self.CountIntervalsStartingAt(start)
                else:
                    msg = "Error: 'start' parameter is missing for COUNT_INTERVALS_STARTING_AT action."
                    
            elif action_code == self.SUM_COUNTS:
                if "counts" in params:
                    counts = params["counts"]
                    msg = self.SumCounts(counts)
                else:
                    msg = "Error: 'counts' parameter is missing for SUM_COUNTS action."
                    
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
    
        Returns the list of building heights in the current environment.
    
        Args:
            None
    
        Returns:
            str: String representation of the building height list.
    
        Example Output:
            "[1, 2, 3, 2]"
        """
        return str(self.heights)

    def CheckIntervalUniqueness(self, start: int, end: int):
        r"""
    
        Checks whether all building heights in the interval from start to end are unique.
    
        Args:
            start (int): Start index of the interval
            end (int): End index of the interval
    
        Returns:
            str: "True" indicates all heights in the interval are unique, "False" indicates there are duplicate heights.
    
        Example Output:
            "True"
        """
        if start < 0 or end >= len(self.heights) or start > end:
            return "False"
            
        seen = set()
        for i in range(start, end + 1):
            if self.heights[i] in seen:
                return "False"
            seen.add(self.heights[i])
        return "True"

    def CountIntervalsStartingAt(self, start: int):
        r"""
    
        Calculates the number of all unique visible intervals starting from the start position.
    
        Args:
            start (int): Starting index
    
        Returns:
            str: The number of unique visible intervals starting from the start position.
    
        Example Output:
            "3"
        """
        if start < 0 or start >= len(self.heights):
            return "0"
            
        count = 0
        seen = set()
        for j in range(start, len(self.heights)):
            if self.heights[j] in seen:
                break
            seen.add(self.heights[j])
            count += 1
        return str(count)

    def SumCounts(self, counts: list):
        r"""
    
        Calculates the sum of multiple counts.
    
        Args:
            counts (list[int]): List of counts
    
        Returns:
            str: The total sum of the counts.
    
        Example Output:
            "6"
        """
        return str(sum(counts))

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 6, Reference answer: 6, Result: Correct, reward=1"
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
        heights_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        heights = ast.literal_eval(heights_str)
        n = len(heights)
        
        counts = []
        for start in range(n):
            count = int(self.step(json.dumps({
                'name': 'CountIntervalsStartingAt',
                'parameters': {'start': start}
            }))[1])
            counts.append(count)
        
        total = int(self.step(json.dumps({
            'name': 'SumCounts',
            'parameters': {'counts': counts}
        }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_heights1 = [1, 2, 3, 2]
    env1 = UniqueVisibleIntervalsEnv.from_env_str(f"UniqueVisibleIntervalsEnv@{{\"heights\": {test_heights1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_heights2 = [5, 5, 5, 5]
    env2 = UniqueVisibleIntervalsEnv.from_env_str(f"UniqueVisibleIntervalsEnv@{{\"heights\": {test_heights2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)