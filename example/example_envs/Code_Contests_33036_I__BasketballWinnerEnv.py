import gymnasium
import ast
import json
from typing import List

class BasketballWinnerEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.PARSE_SCORE = 0
        self.ADD_TO_TEAM_SCORE = 1
        self.COMPARE_SCORES = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "ParseScore": self.PARSE_SCORE,
            "AddToTeamScore": self.ADD_TO_TEAM_SCORE,
            "CompareScores": self.COMPARE_SCORES,
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
        prefix = "BasketballWinnerEnv@"
        if not env_str.startswith(prefix):
            return None
        return BasketballWinnerEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.scores = options.get("scores", [])
        self.team_a_score = 0
        self.team_b_score = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        score_a = 0
        score_b = 0

        for score in self.scores:
            if score[-1] == 'A':
                score_a += int(score[:-1])
            elif score[-1] == 'B':
                score_b += int(score[:-1])
        
        if score_a > score_b:
            return 'A'
        elif score_b > score_a:
            return 'B'
        else:
            return 'D'

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
            
            if action_code == self.PARSE_SCORE:
                if "score_str" in params:
                    score_str = params["score_str"]
                    msg = self.ParseScore(score_str)
                else:
                    msg = "Error: 'score_str' parameter is missing for PARSE_SCORE action."
            
            elif action_code == self.ADD_TO_TEAM_SCORE:
                if "team" in params and "score" in params:
                    team = params["team"]
                    score = params["score"]
                    msg = self.AddToTeamScore(team, score)
                else:
                    msg = "Error: 'team' or 'score' parameter is missing for ADD_TO_TEAM_SCORE action."
                    
            elif action_code == self.COMPARE_SCORES:
                if "score_a" in params and "score_b" in params:
                    score_a = params["score_a"]
                    score_b = params["score_b"]
                    msg = self.CompareScores(score_a, score_b)
                else:
                    msg = "Error: 'score_a' or 'score_b' parameter is missing for COMPARE_SCORES action."
                    
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
    def ParseScore(self, score_str: str):
        r"""
    
        Parse a single score string to extract the score and the corresponding team.
    
        Args:
            score_str (str): The score string in the format of "number+A/B", e.g., "10A".
    
        Returns:
            str: A JSON string containing the score and the team.
    
        Example Output:
            "{\"score\": 10, \"team\": \"A\"}"
        """
        team = score_str[-1]
        score = int(score_str[:-1])
        return json.dumps({"score": score, "team": team})

    def AddToTeamScore(self, team: str, score: int):
        r"""
    
        Add the score to the total score of the corresponding team and return the updated total score of the team.
    
        Args:
            team (str): The team identifier, "A" or "B".
            score (int): The score to be added.
    
        Returns:
            str: A JSON string containing the team and the updated total score.
    
        Example Output:
            "{\"team\": \"A\", \"total_score\": 35}"
        """
        if team == "A":
            self.team_a_score += score
            return json.dumps({"team": "A", "total_score": self.team_a_score})
        elif team == "B":
            self.team_b_score += score
            return json.dumps({"team": "B", "total_score": self.team_b_score})
        else:
            return json.dumps({"error": "Invalid team"})

    def CompareScores(self, score_a: int, score_b: int):
        r"""
    
        Compare the total scores of Team A and Team B to determine the match result.
    
        Args:
            score_a (int): The total score of Team A.
            score_b (int): The total score of Team B.
    
        Returns:
            str: The match result, "A" indicates Team A wins, "B" indicates Team B wins, and "D" indicates a draw.
    
        Example Output:
            "B"
        """
        if score_a > score_b:
            return "A"
        elif score_b > score_a:
            return "B"
        else:
            return "D"

    def Observe(self):
        r"""
    
        Obtain the current environment state, including the list of score records and the current scores of both teams.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the current environment state.
    
        Example Output:
            "{\"scores\": [\"10A\", \"20B\"], \"team_a_score\": 10, \"team_b_score\": 20}"
        """
        return json.dumps({
            "scores": self.scores,
            "team_a_score": self.team_a_score,
            "team_b_score": self.team_b_score
        })

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer submitted by the user, "A", "B", or "D".
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: B, Reference answer: B, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
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
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        observe_data = json.loads(observe_result)
        scores_list = observe_data['scores']
        
        for score_str in scores_list:
            parse_result = self.step(json.dumps({'name': 'ParseScore', 'parameters': {'score_str': score_str}}))[1]
            parsed_data = json.loads(parse_result)
            team = parsed_data['team']
            score = parsed_data['score']
            self.step(json.dumps({'name': 'AddToTeamScore', 'parameters': {'team': team, 'score': score}}))
        
        final_observe = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        final_data = json.loads(final_observe)
        score_a = final_data['team_a_score']
        score_b = final_data['team_b_score']
        
        result = self.step(json.dumps({'name': 'CompareScores', 'parameters': {'score_a': score_a, 'score_b': score_b}}))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_scores1 = ["10A", "20B", "15A", "25B"]
    env1 = BasketballWinnerEnv.from_env_str(f"BasketballWinnerEnv@{{\"scores\": {test_scores1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_scores2 = ["30A", "10B", "20A", "15B"]
    env2 = BasketballWinnerEnv.from_env_str(f"BasketballWinnerEnv@{{\"scores\": {test_scores2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_scores3 = ["10A", "25A", "20B", "15B"]
    env3 = BasketballWinnerEnv.from_env_str(f"BasketballWinnerEnv@{{\"scores\": {test_scores3}}}")
    print(env3.solve())
    print("step count:", env3.step_count)