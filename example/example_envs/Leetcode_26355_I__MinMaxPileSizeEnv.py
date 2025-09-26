# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import heapq
import ast
import json
from typing import List

class MinMaxPileSizeEnv(gymnasium.Env):
    def __init__(self, env_str: str = None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.SPLIT_PILE = 1
        self.GET_CURRENT_MAX = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "SplitPile": self.SPLIT_PILE,
            "GetCurrentMax": self.GET_CURRENT_MAX,
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
        prefix = "MinMaxPileSizeEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinMaxPileSizeEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options: dict = {}):
        self.nums = options.get("nums", [])
        self.current_piles = self.nums.copy()
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self) -> int:
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.nums:
            return 0
            
        max_heap = [-x for x in self.nums]
        heapq.heapify(max_heap)
        
        while -max_heap[0] > 1:
            largest = -heapq.heappop(max_heap)
            
            # Split it into two possible smaller piles
            first_half = largest // 2
            second_half = largest - first_half
            
            heapq.heappush(max_heap, -first_half)
            heapq.heappush(max_heap, -second_half)
        
        return -max_heap[0]

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
                
            elif action_code == self.SPLIT_PILE:
                if "index" in params and "split_size" in params:
                    index = params["index"]
                    split_size = params["split_size"]
                    msg = self.SplitPile(index, split_size)
                else:
                    msg = "Error: 'index' or 'split_size' parameter is missing for SPLIT_PILE action."
                    
            elif action_code == self.GET_CURRENT_MAX:
                msg = self.GetCurrentMax()
                
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
    def Observe(self) -> str:
        r"""
    
        Obtain the current state of all stone piles.
    
        Args:
            None
    
        Returns:
            str: A list of the quantities of all current stone piles.
    
        Example Output:
            "[3, 2, 5]"
        """
        return json.dumps(self.current_piles)

    def SplitPile(self, index: int, split_size: int) -> str:
        r"""
    
        Split the stone pile at the specified index into two smaller piles.
    
        Args:
            index (int): The index of the pile to be split.
            split_size (int): The size of one of the piles after splitting, which must be greater than 0 and smaller than the size of the original pile.
    
        Returns:
            str: A list of the quantities of all stone piles after splitting.
    
        Example Output:
            "[2, 2, 3]"
        """
        if index < 0 or index >= len(self.current_piles):
            return "Error: Invalid pile index."
            
        pile_size = self.current_piles[index]
        if pile_size < 2:
            return "Error: Cannot split a pile with less than 2 stones."
            
        if split_size <= 0 or split_size >= pile_size:
            return "Error: Split size must be between 1 and pile size - 1."
            
        # Remove the original pile and add the two new piles
        self.current_piles.pop(index)
        self.current_piles.append(split_size)
        self.current_piles.append(pile_size - split_size)
        
        return json.dumps(self.current_piles)

    def GetCurrentMax(self) -> str:
        r"""
    
        Obtain the size of the current largest stone pile.
    
        Args:
            None
    
        Returns:
            str: The size of the current largest stone pile.
    
        Example Output:
            "5"
        """
        if not self.current_piles:
            return "0"
        return str(max(self.current_piles))

    def Done(self, answer: int) -> str:
        r"""
    
        Submit whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The minimum possible maximum pile size submitted by the user.
    
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
        initial_piles = ast.literal_eval(self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1])
        current_max = int(self.step(json.dumps({'name': 'GetCurrentMax', 'parameters': {}}))[1])
        
        while True:
            piles = ast.literal_eval(self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1])
            split_done = False
            for i in range(len(piles)):
                pile = piles[i]
                if pile > current_max:
                    pass
                if pile >= 2:
                    a = pile // 2
                    b = pile - a
                    new_max_candidate = max(current_max, a, b)
                    if pile == current_max:
                        new_possible_max = max(a, b)
                        if new_possible_max < current_max:
                            self.step(json.dumps({'name': 'SplitPile', 'parameters': {'index': i, 'split_size': a}}))
                            current_max = new_possible_max
                            split_done = True
                            break
            if not split_done:
                break
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': current_max}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_nums1 = [4, 3]
    env1 = MinMaxPileSizeEnv.from_env_str(f"MinMaxPileSizeEnv@{{\"nums\": {test_nums1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_nums2 = [2, 7, 9]
    env2 = MinMaxPileSizeEnv.from_env_str(f"MinMaxPileSizeEnv@{{\"nums\": {test_nums2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)