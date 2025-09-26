# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MinEnergyPathEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_DP = 0
        self.FILL_FIRST_ROW = 1
        self.FILL_FIRST_COLUMN = 2
        self.FILL_DP_TABLE_CELL = 3
        self.GET_FINAL_RESULT = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeDP": self.INITIALIZE_DP,
            "FillFirstRow": self.FILL_FIRST_ROW,
            "FillFirstColumn": self.FILL_FIRST_COLUMN,
            "FillDPTableCell": self.FILL_DP_TABLE_CELL,
            "GetFinalResult": self.GET_FINAL_RESULT,
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
        prefix = "MinEnergyPathEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinEnergyPathEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.grid = options.get("grid", [])
        self.n = len(self.grid)
        self.m = len(self.grid[0]) if self.n > 0 else 0
        self.dp = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.grid or len(self.grid) == 0 or len(self.grid[0]) == 0:
            return 0
            
        n = len(self.grid)
        m = len(self.grid[0])
        
        dp = [[0] * m for _ in range(n)]
        dp[0][0] = self.grid[0][0]
        
        for j in range(1, m):
            dp[0][j] = dp[0][j-1] + self.grid[0][j]
            
        for i in range(1, n):
            dp[i][0] = dp[i-1][0] + self.grid[i][0]
            
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + self.grid[i][j]
        
        return dp[n-1][m-1]

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
            
            if action_code == self.INITIALIZE_DP:
                msg = self.InitializeDP()
            
            elif action_code == self.FILL_FIRST_ROW:
                msg = self.FillFirstRow()
                
            elif action_code == self.FILL_FIRST_COLUMN:
                msg = self.FillFirstColumn()
                
            elif action_code == self.FILL_DP_TABLE_CELL:
                if "i" in params and "j" in params:
                    i = params["i"]
                    j = params["j"]
                    msg = self.FillDPTableCell(i, j)
                else:
                    msg = "Error: 'i' or 'j' parameter is missing for FILL_DP_TABLE_CELL action."
                    
            elif action_code == self.GET_FINAL_RESULT:
                msg = self.GetFinalResult()
                
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
    def InitializeDP(self):
        r"""
    
        Initialize the DP table, set the size of the DP table, and set the value of the starting cell to the value of the corresponding cell in the grid.
    
        Args:
            None
    
        Returns:
            str: Information of the initialized DP table.
    
        Example Output:
            "[[1, 0, 0], [0, 0, 0], [0, 0, 0]]"
        """
        self.dp = [[0] * self.m for _ in range(self.n)]
        if self.n > 0 and self.m > 0:
            self.dp[0][0] = self.grid[0][0]
        return json.dumps(self.dp)

    def FillFirstRow(self):
        r"""
    
        Fill the first row of the DP table, where the value of each cell is the value of the previous cell plus the value of the current grid cell.
    
        Args:
            None
    
        Returns:
            str: Information of the filled first row of the DP table.
    
        Example Output:
            "[1, 4, 5]"
        """
        if self.m <= 1:
            return json.dumps(self.dp[0])
            
        for j in range(1, self.m):
            self.dp[0][j] = self.dp[0][j-1] + self.grid[0][j]
        return json.dumps(self.dp[0])

    def FillFirstColumn(self):
        r"""
    
        Fill the first column of the DP table, where the value of each cell is the value of the previous cell (above) plus the value of the current grid cell.
    
        Args:
            None
    
        Returns:
            str: Information of the filled first column of the DP table.
    
        Example Output:
            "[1, 2, 6]"
        """
        if self.n <= 1:
            return json.dumps([self.dp[i][0] for i in range(self.n)])
            
        for i in range(1, self.n):
            self.dp[i][0] = self.dp[i-1][0] + self.grid[i][0]
        return json.dumps([self.dp[i][0] for i in range(self.n)])

    def FillDPTableCell(self, i: int, j: int):
        r"""
    
        Fill the cell at the specified position (i, j) in the DP table, with the value being the minimum value of the cell above or to the left plus the value of the current grid cell.
    
        Args:
            i (int): Row index of the cell
            j (int): Column index of the cell
    
        Returns:
            str: Information of the filled cell's value and position.
    
        Example Output:
            "DP[1][1] = 7"
        """
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            return f"Error: Invalid cell coordinates ({i}, {j})"
            
        if i == 0 and j == 0:
            return f"DP[{i}][{j}] = {self.dp[i][j]}"
            
        self.dp[i][j] = min(self.dp[i-1][j], self.dp[i][j-1]) + self.grid[i][j]
        return f"DP[{i}][{j}] = {self.dp[i][j]}"

    def GetFinalResult(self):
        r"""
    
        Get the value of the cell at the bottom-right corner of the DP table, which is the minimum energy required to reach the end.
    
        Args:
            None
    
        Returns:
            str: The minimum energy required to reach the end.
    
        Example Output:
            "7"
        """
        if not self.dp or self.n == 0 or self.m == 0:
            return "0"
        return str(self.dp[self.n-1][self.m-1])

    def Observe(self):
        r"""
    
        Get the information of the current grid.
    
        Args:
            None
    
        Returns:
            str: Information of the current grid.
    
        Example Output:
            "[[1, 3, 1], [1, 5, 1], [4, 2, 1]]"
        """
        return json.dumps(self.grid)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The minimum energy value submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 7, Reference answer: 7, Result: Correct, reward=1"
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
        self.step(json.dumps({'name': 'InitializeDP', 'parameters': {}}))
        self.step(json.dumps({'name': 'FillFirstRow', 'parameters': {}}))
        self.step(json.dumps({'name': 'FillFirstColumn', 'parameters': {}}))
        grid_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        grid = ast.literal_eval(grid_str)
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        for i in range(1, n):
            for j in range(1, m):
                self.step(json.dumps({'name': 'FillDPTableCell', 'parameters': {'i': i, 'j': j}}))
        min_energy = int(self.step(json.dumps({'name': 'GetFinalResult', 'parameters': {}}))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': min_energy}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_grid1 = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]
    env1 = MinEnergyPathEnv.from_env_str(f"MinEnergyPathEnv@{{\"grid\": {test_grid1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_grid2 = [
        [2, 1],
        [1, 3]
    ]
    env2 = MinEnergyPathEnv.from_env_str(f"MinEnergyPathEnv@{{\"grid\": {test_grid2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)