# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from collections import deque

class PathFindingEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_START_END = 0
        self.GET_NEXT_CELL = 1
        self.GET_NEIGHBORS = 2
        self.CHECK_NEIGHBOR_VALIDITY = 3
        self.ADD_TO_QUEUE_VISITED = 4
        self.CHECK_TARGET_REACHED = 5
        self.OBSERVE = 6
        self.DONE = 7

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckStartEnd": self.CHECK_START_END,
            "GetNextCell": self.GET_NEXT_CELL,
            "GetNeighbors": self.GET_NEIGHBORS,
            "CheckNeighborValidity": self.CHECK_NEIGHBOR_VALIDITY,
            "AddToQueueVisited": self.ADD_TO_QUEUE_VISITED,
            "CheckTargetReached": self.CHECK_TARGET_REACHED,
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
        prefix = "PathFindingEnv@"
        if not env_str.startswith(prefix):
            return None
        return PathFindingEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.grid = options.get("grid", [])
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.queue = deque()
        self.visited = set()
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if self.rows == 0 or self.cols == 0:
            return "NO"
            
        if self.grid[0][0] == '1' or self.grid[self.rows-1][self.cols-1] == '1':
            return "NO"
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        queue = deque([(0, 0)])
        visited = set((0, 0))
        
        while queue:
            x, y = queue.popleft()
            if x == self.rows - 1 and y == self.cols - 1:
                return "YES"
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] == '0' and (nx, ny) not in visited:
                    queue.append((nx, ny))
                    visited.add((nx, ny))

        return "NO"

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
            
            if action_code == self.CHECK_START_END:
                msg = self.CheckStartEnd()
            
            elif action_code == self.GET_NEXT_CELL:
                msg = self.GetNextCell()
                
            elif action_code == self.GET_NEIGHBORS:
                if "x" in params and "y" in params:
                    x = params["x"]
                    y = params["y"]
                    msg = self.GetNeighbors(x, y)
                else:
                    msg = "Error: 'x' or 'y' parameter is missing for GET_NEIGHBORS action."
                    
            elif action_code == self.CHECK_NEIGHBOR_VALIDITY:
                if "nx" in params and "ny" in params:
                    nx = params["nx"]
                    ny = params["ny"]
                    msg = self.CheckNeighborValidity(nx, ny)
                else:
                    msg = "Error: 'nx' or 'ny' parameter is missing for CHECK_NEIGHBOR_VALIDITY action."
                    
            elif action_code == self.ADD_TO_QUEUE_VISITED:
                if "nx" in params and "ny" in params:
                    nx = params["nx"]
                    ny = params["ny"]
                    msg = self.AddToQueueVisited(nx, ny)
                else:
                    msg = "Error: 'nx' or 'ny' parameter is missing for ADD_TO_QUEUE_VISITED action."
                    
            elif action_code == self.CHECK_TARGET_REACHED:
                if "x" in params and "y" in params:
                    x = params["x"]
                    y = params["y"]
                    msg = self.CheckTargetReached(x, y)
                else:
                    msg = "Error: 'x' or 'y' parameter is missing for CHECK_TARGET_REACHED action."
                    
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
    def CheckStartEnd(self):
        r"""
    
        Check if the start point (0,0) and end point (rows-1, cols-1) are passable.
    
        Args:
            None
    
        Returns:
            str: "valid" if both the start and end points are passable, "invalid" otherwise.
    
        Example Output:
            "valid"
        """
        if self.grid[0][0] == '1' or self.grid[self.rows-1][self.cols-1] == '1':
            return "invalid"
        return "valid"

    def GetNextCell(self):
        r"""
    
        Get the next cell to process from the queue.
    
        Args:
            None
    
        Returns:
            str: The coordinates of the next cell in the format "x,y", or "empty" if the queue is empty.
    
        Example Output:
            "0,0"
        """
        if not self.queue:
            return "empty"
        x, y = self.queue.popleft()
        return f"{x},{y}"

    def GetNeighbors(self, x: int, y: int):
        r"""
    
        Get the coordinates of the four-directional neighbors of the specified cell (x,y).
    
        Args:
            x (int): Row index of the current cell.
            y (int): Column index of the current cell.
    
        Returns:
            str: The coordinates of the four neighbors in the format "x1,y1;x2,y2;x3,y3;x4,y4".
    
        Example Output:
            "0,1;1,0;0,-1;-1,0"
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbors.append(f"{nx},{ny}")
        return ";".join(neighbors)

    def CheckNeighborValidity(self, nx: int, ny: int):
        r"""
    
        Check if the neighbor cell (nx, ny) is valid: within the grid range, passable, and not visited.
    
        Args:
            nx (int): Row index of the neighbor cell.
            ny (int): Column index of the neighbor cell.
    
        Returns:
            str: "valid" if the neighbor is valid, "invalid" otherwise.
    
        Example Output:
            "valid"
        """
        if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] == '0' and (nx, ny) not in self.visited:
            return "valid"
        return "invalid"

    def AddToQueueVisited(self, nx: int, ny: int):
        r"""
    
        Add the valid neighbor cell to the queue and the visited set.
    
        Args:
            nx (int): Row index of the neighbor cell.
            ny (int): Column index of the neighbor cell.
    
        Returns:
            str: Prompt message for successful addition.
    
        Example Output:
            "Added (1,0) to queue and visited"
        """
        self.queue.append((nx, ny))
        self.visited.add((nx, ny))
        return f"Added ({nx},{ny}) to queue and visited"

    def CheckTargetReached(self, x: int, y: int):
        r"""
    
        Check if the current cell (x,y) is the end point (rows-1, cols-1).
    
        Args:
            x (int): Row index of the current cell.
            y (int): Column index of the current cell.
    
        Returns:
            str: "yes" if it is the end point, "no" otherwise.
    
        Example Output:
            "no"
        """
        if x == self.rows - 1 and y == self.cols - 1:
            return "yes"
        return "no"

    def Observe(self):
        r"""
    
        Return observation information about the current environment state.
    
        Args:
            None
    
        Returns:
            str: Information describing the current queue state and the number of visited cells.
    
        Example Output:
            "Queue size: 3, Visited cells: 5"
        """
        return f"Queue size: {len(self.queue)}, Visited cells: {len(self.visited)}"

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return result information.
    
        Args:
            answer (str): The answer submitted by the user, which should be "YES" or "NO".
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: YES, Reference answer: YES, Result: Correct, reward=1"
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
        start_end_check = self.step(json.dumps({'name': 'CheckStartEnd', 'parameters': {}}))[1]
        if start_end_check != "valid":
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'NO'}}))[1]
        
        first_cell = self.step(json.dumps({'name': 'GetNextCell', 'parameters': {}}))[1]
        if first_cell == "empty":
            self.step(json.dumps({'name': 'AddToQueueVisited', 'parameters': {'nx': 0, 'ny': 0}}))
        
        while True:
            current_cell = self.step(json.dumps({'name': 'GetNextCell', 'parameters': {}}))[1]
            if current_cell == "empty":
                return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'NO'}}))[1]
            
            x, y = map(int, current_cell.split(','))
            
            target_check = self.step(json.dumps({'name': 'CheckTargetReached', 'parameters': {'x': x, 'y': y}}))[1]
            if target_check == "yes":
                return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 'YES'}}))[1]
            
            neighbors_str = self.step(json.dumps({'name': 'GetNeighbors', 'parameters': {'x': x, 'y': y}}))[1]
            neighbors = neighbors_str.split(';')
            
            for neighbor in neighbors:
                nx, ny = map(int, neighbor.split(','))
                validity = self.step(json.dumps({'name': 'CheckNeighborValidity', 'parameters': {'nx': nx, 'ny': ny}}))[1]
                if validity == "valid":
                    self.step(json.dumps({'name': 'AddToQueueVisited', 'parameters': {'nx': nx, 'ny': ny}}))
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    grid1 = ["0001", "0110", "0100", "0000"]
    env1 = PathFindingEnv.from_env_str(f"PathFindingEnv@{{\"grid\": {grid1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    grid2 = ["010", "011", "100"]
    env2 = PathFindingEnv.from_env_str(f"PathFindingEnv@{{\"grid\": {grid2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)