# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class GameWinnerDetermineEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_ROUND_SCORE = 0
        self.DETERMINE_WINNER = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateRoundScore": self.CALCULATE_ROUND_SCORE,
            "DetermineWinner": self.DETERMINE_WINNER,
            "Observe": self.OBSERVE,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property of the environment
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "GameWinnerDetermineEnv@"
        if not env_str.startswith(prefix):
            return None
        return GameWinnerDetermineEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 0)
        self.player1 = options.get("player1", [])
        self.player2 = options.get("player2", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        score1, score2 = 0, 0
        
        for i in range(self.n):
            if self.player1[i] > self.player2[i]:
                score1 += self.player1[i] - self.player2[i]
            elif self.player2[i] > self.player1[i]:
                score2 += self.player2[i] - self.player1[i]
        
        if score1 > score2:
            return "PLAYER 1 WINS"
        elif score2 > score1:
            return "PLAYER 2 WINS"
        else:
            return "TIE"

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
            
            if action_code == self.CALCULATE_ROUND_SCORE:
                if "round_index" in params and "p1_choice" in params and "p2_choice" in params:
                    round_index = params["round_index"]
                    p1_choice = params["p1_choice"]
                    p2_choice = params["p2_choice"]
                    msg = self.CalculateRoundScore(round_index, p1_choice, p2_choice)
                else:
                    msg = "Error: 'round_index', 'p1_choice' or 'p2_choice' parameter is missing for CALCULATE_ROUND_SCORE action."
            
            elif action_code == self.DETERMINE_WINNER:
                if "score1" in params and "score2" in params:
                    score1 = params["score1"]
                    score2 = params["score2"]
                    msg = self.DetermineWinner(score1, score2)
                else:
                    msg = "Error: 'score1' or 'score2' parameter is missing for DETERMINE_WINNER action."
                    
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
    def CalculateRoundScore(self, round_index: int, p1_choice: int, p2_choice: int):
        r"""
    
        Calculate the score changes of the two players in a single round.
    
        Args:
            round_index (int): Round index (starting from 0).
            p1_choice (int): Player 1's choice in this round.
            p2_choice (int): Player 2's choice in this round.
    
        Returns:
            str: A JSON string containing the round index and the score changes in this round.
    
        Example Output:
            "{\"round_index\": 0, \"p1_score_change\": 1, \"p2_score_change\": 0}"
        """
        p1_change = 0
        p2_change = 0
        
        if p1_choice > p2_choice:
            p1_change = p1_choice - p2_choice
        elif p2_choice > p1_choice:
            p2_change = p2_choice - p1_choice
            
        result = {
            "round_index": round_index,
            "p1_score_change": p1_change,
            "p2_score_change": p2_change
        }
        return json.dumps(result)

    def DetermineWinner(self, score1: int, score2: int):
        r"""
    
        Determine the game result based on the total scores of the two players.
    
        Args:
            score1 (int): Player 1's total score.
            score2 (int): Player 2's total score.
    
        Returns:
            str: The game result, which may be "PLAYER 1 WINS", "PLAYER 2 WINS", or "TIE".
    
        Example Output:
            "PLAYER 1 WINS"
        """
        if score1 > score2:
            return "PLAYER 1 WINS"
        elif score2 > score1:
            return "PLAYER 2 WINS"
        else:
            return "TIE"

    def Observe(self):
        r"""
    
        Obtain basic game information in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the number of rounds and the list of players' choices.
    
        Example Output:
            "{\"n\": 3, \"player1\": [4, 5, 8], \"player2\": [3, 5, 7]}"
        """
        observation = {
            "n": self.n,
            "player1": self.player1,
            "player2": self.player2
        }
        return json.dumps(observation)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (str): The answer submitted by the user, which should be "PLAYER 1 WINS", "PLAYER 2 WINS", or "TIE".
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: PLAYER 1 WINS, Reference answer: PLAYER 1 WINS, Result: Correct, reward=1"
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
        game_info = json.loads(observe_result)
        n = game_info['n']
        player1_choices = game_info['player1']
        player2_choices = game_info['player2']
        
        total_p1 = 0
        total_p2 = 0
        
        for round_index in range(n):
            p1_choice = player1_choices[round_index]
            p2_choice = player2_choices[round_index]
            round_score_json = self.step(json.dumps({
                'name': 'CalculateRoundScore',
                'parameters': {
                    'round_index': round_index,
                    'p1_choice': p1_choice,
                    'p2_choice': p2_choice
                }
            }))[1]
            round_score = json.loads(round_score_json)
            total_p1 += round_score['p1_score_change']
            total_p2 += round_score['p2_score_change']
        
        winner = self.step(json.dumps({
            'name': 'DetermineWinner',
            'parameters': {
                'score1': total_p1,
                'score2': total_p2
            }
        }))[1]
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': winner}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env1 = GameWinnerDetermineEnv.from_env_str(
        "GameWinnerDetermineEnv@{\"n\": 3, \"player1\": [4, 5, 8], \"player2\": [3, 5, 7]}"
    )
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    env2 = GameWinnerDetermineEnv.from_env_str(
        "GameWinnerDetermineEnv@{\"n\": 2, \"player1\": [4, 6], \"player2\": [5, 6]}"
    )
    print(env2.solve())
    print("step count:", env2.step_count)