import gymnasium
import ast
import json
import random

class AdjustedAverageEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GET_SCORE_COUNT = 0
        self.GET_MIN_SCORE = 1
        self.GET_MAX_SCORE = 2
        self.REMOVE_SCORE = 3
        self.SUM_SCORES = 4
        self.CALCULATE_AVERAGE = 5
        self.ROUND_TO_ONE_DECIMAL = 6
        self.OBSERVE = 7
        self.DONE = 8

        # [Required] Define the action mapping
        self.func_mapping = {
            "GetScoreCount": self.GET_SCORE_COUNT,
            "GetMinScore": self.GET_MIN_SCORE,
            "GetMaxScore": self.GET_MAX_SCORE,
            "RemoveScore": self.REMOVE_SCORE,
            "SumScores": self.SUM_SCORES,
            "CalculateAverage": self.CALCULATE_AVERAGE,
            "RoundToOneDecimal": self.ROUND_TO_ONE_DECIMAL,
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
        prefix = "AdjustedAverageEnv@"
        if not env_str.startswith(prefix):
            return None
        return AdjustedAverageEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.scores = options.get("scores", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if len(self.scores) < 3:
            return 0.0
        
        temp_scores = self.scores.copy()
        min_score = min(temp_scores)
        max_score = max(temp_scores)
        
        temp_scores.remove(min_score)
        temp_scores.remove(max_score)
        
        adjusted_average = sum(temp_scores) / len(temp_scores)
        return round(adjusted_average, 1)

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
            msg = ""
            
            if action_code == self.GET_SCORE_COUNT:
                msg = self.GetScoreCount()
            
            elif action_code == self.GET_MIN_SCORE:
                if "scores" in params:
                    scores = params["scores"]
                    msg = self.GetMinScore(scores)
                else:
                    msg = "Error: 'scores' parameter is missing for GET_MIN_SCORE action."
            
            elif action_code == self.GET_MAX_SCORE:
                if "scores" in params:
                    scores = params["scores"]
                    msg = self.GetMaxScore(scores)
                else:
                    msg = "Error: 'scores' parameter is missing for GET_MAX_SCORE action."
            
            elif action_code == self.REMOVE_SCORE:
                if "scores" in params and "value" in params:
                    scores = params["scores"]
                    value = params["value"]
                    msg = self.RemoveScore(scores, value)
                else:
                    msg = "Error: 'scores' or 'value' parameter is missing for REMOVE_SCORE action."
            
            elif action_code == self.SUM_SCORES:
                if "scores" in params:
                    scores = params["scores"]
                    msg = self.SumScores(scores)
                else:
                    msg = "Error: 'scores' parameter is missing for SUM_SCORES action."
            
            elif action_code == self.CALCULATE_AVERAGE:
                if "sum_scores" in params and "count" in params:
                    sum_scores = params["sum_scores"]
                    count = params["count"]
                    msg = self.CalculateAverage(sum_scores, count)
                else:
                    msg = "Error: 'sum_scores' or 'count' parameter is missing for CALCULATE_AVERAGE action."
            
            elif action_code == self.ROUND_TO_ONE_DECIMAL:
                if "value" in params:
                    value = params["value"]
                    msg = self.RoundToOneDecimal(value)
                else:
                    msg = "Error: 'value' parameter is missing for ROUND_TO_ONE_DECIMAL action."
            
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
    def GetScoreCount(self):
        r"""
    
        Get the length of the score list.
    
        Args:
            None
    
        Returns:
            str: The length of the score list.
    
        Example Output:
            "5"
        """
        return str(len(self.scores))

    def GetMinScore(self, scores: list):
        r"""
    
        Get the minimum score in the score list.
    
        Args:
            scores (list[int]): The score list.
    
        Returns:
            str: The minimum score in the score list.
    
        Example Output:
            "1"
        """
        return str(min(scores))

    def GetMaxScore(self, scores: list):
        r"""
    
        Get the maximum score in the score list.
    
        Args:
            scores (list[int]): The score list.
    
        Returns:
            str: The maximum score in the score list.
    
        Example Output:
            "5"
        """
        return str(max(scores))

    def RemoveScore(self, scores: list, value: int):
        r"""
    
        Remove the first occurrence of the specified value from the score list and return the new list.
    
        Args:
            scores (list[int]): The original score list.
            value (int): The score value to be removed.
    
        Returns:
            str: The JSON string of the new list after removing the specified value.
    
        Example Output:
            "[2, 3, 4, 5]"
        """
        new_scores = scores.copy()
        new_scores.remove(value)
        return json.dumps(new_scores)

    def SumScores(self, scores: list):
        r"""
    
        Calculate the sum of all scores in the score list.
    
        Args:
            scores (list[int]): The score list.
    
        Returns:
            str: The total sum of the scores.
    
        Example Output:
            "9"
        """
        return str(sum(scores))

    def CalculateAverage(self, sum_scores: float, count: int):
        r"""
    
        Calculate the average based on the total score sum and the number of scores.
    
        Args:
            sum_scores (float): The total sum of the scores.
            count (int): The number of scores.
    
        Returns:
            str: The calculated average.
    
        Example Output:
            "3.0"
        """
        if count == 0:
            return "0.0"
        return str(sum_scores / count)

    def RoundToOneDecimal(self, value: float):
        r"""
    
        Round the value to one decimal place.
    
        Args:
            value (float): The value to be rounded.
    
        Returns:
            str: The value rounded to one decimal place.
    
        Example Output:
            "3.0"
        """
        return str(round(value, 1))

    def Observe(self):
        r"""
    
        Get the score list in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current score list.
    
        Example Output:
            "[1, 2, 3, 4, 5]"
        """
        return json.dumps(self.scores)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (float): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: 3.0, Reference answer: 3.0, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = abs(answer - ref_answer) < 0.01
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
        scores_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        scores = ast.literal_eval(scores_str)
        
        count_str = self.step(json.dumps({'name': 'GetScoreCount', 'parameters': {}}))[1]
        score_count = int(count_str)
        
        if score_count < 3:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0.0}}))[1]
        
        min_score = int(self.step(json.dumps({'name': 'GetMinScore', 'parameters': {'scores': scores}}))[1])
        max_score = int(self.step(json.dumps({'name': 'GetMaxScore', 'parameters': {'scores': scores}}))[1])
        
        scores_after_remove_min = ast.literal_eval(self.step(json.dumps({'name': 'RemoveScore', 'parameters': {'scores': scores, 'value': min_score}}))[1])
        scores_adjusted = ast.literal_eval(self.step(json.dumps({'name': 'RemoveScore', 'parameters': {'scores': scores_after_remove_min, 'value': max_score}}))[1])
        
        sum_str = self.step(json.dumps({'name': 'SumScores', 'parameters': {'scores': scores_adjusted}}))[1]
        sum_scores = float(sum_str)
        
        adjusted_count = len(scores_adjusted)
        
        average_str = self.step(json.dumps({'name': 'CalculateAverage', 'parameters': {'sum_scores': sum_scores, 'count': adjusted_count}}))[1]
        average = float(average_str)
        
        rounded_average_str = self.step(json.dumps({'name': 'RoundToOneDecimal', 'parameters': {'value': average}}))[1]
        rounded_average = float(rounded_average_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': rounded_average}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_scores1 = [1, 2, 3, 4, 5]
    env1 = AdjustedAverageEnv.from_env_str(f"AdjustedAverageEnv@{{\"scores\": {test_scores1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_scores2 = [3, 4]
    env2 = AdjustedAverageEnv.from_env_str(f"AdjustedAverageEnv@{{\"scores\": {test_scores2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)

    # test case 3 (random test)
    print("\nTest Case 3 (Random):")
    test_scores3 = [random.randint(1, 5) for _ in range(random.randint(0, 10))]
    print(f"Random scores: {test_scores3}")
    env3 = AdjustedAverageEnv.from_env_str(f"AdjustedAverageEnv@{{\"scores\": {test_scores3}}}")
    print(env3.solve())
    print("step count:", env3.step_count)