# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json
from collections import defaultdict

class MaxSubarrayWithKDistinctEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.INITIALIZE_SLIDING_WINDOW = 1
        self.MOVE_RIGHT_POINTER = 2
        self.ADJUST_LEFT_POINTER = 3
        self.UPDATE_MAX_LENGTH = 4
        self.CHECK_COMPLETION = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "InitializeSlidingWindow": self.INITIALIZE_SLIDING_WINDOW,
            "MoveRightPointer": self.MOVE_RIGHT_POINTER,
            "AdjustLeftPointer": self.ADJUST_LEFT_POINTER,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
            "CheckCompletion": self.CHECK_COMPLETION,
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
        prefix = "MaxSubarrayWithKDistinctEnv@"
        if not env_str.startswith(prefix):
            return None
        return MaxSubarrayWithKDistinctEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.arr = options.get("arr", [])
        self.k = options.get("k", 0)
        self.window_state = None  # To store sliding window state
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.arr)
        if n == 0 or self.k == 0:
            return 0

        left = 0
        right = 0
        max_length = 0
        unique_count = 0
        freq_map = defaultdict(int)

        while right < n:
            if freq_map[self.arr[right]] == 0:
                unique_count += 1
            freq_map[self.arr[right]] += 1
            right += 1

            while unique_count > self.k:
                freq_map[self.arr[left]] -= 1
                if freq_map[self.arr[left]] == 0:
                    unique_count -= 1
                left += 1
            
            if unique_count == self.k:
                max_length = max(max_length, right - left)

        return max_length

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
            
            elif action_code == self.INITIALIZE_SLIDING_WINDOW:
                msg = self.InitializeSlidingWindow()
            
            elif action_code == self.MOVE_RIGHT_POINTER:
                if "window_state" in params:
                    window_state = params["window_state"]
                    msg = self.MoveRightPointer(window_state)
                else:
                    msg = "Error: 'window_state' parameter is missing for MOVE_RIGHT_POINTER action."
            
            elif action_code == self.ADJUST_LEFT_POINTER:
                if "window_state" in params:
                    window_state = params["window_state"]
                    msg = self.AdjustLeftPointer(window_state)
                else:
                    msg = "Error: 'window_state' parameter is missing for ADJUST_LEFT_POINTER action."
            
            elif action_code == self.UPDATE_MAX_LENGTH:
                if "window_state" in params:
                    window_state = params["window_state"]
                    msg = self.UpdateMaxLength(window_state)
                else:
                    msg = "Error: 'window_state' parameter is missing for UPDATE_MAX_LENGTH action."
            
            elif action_code == self.CHECK_COMPLETION:
                if "window_state" in params:
                    window_state = params["window_state"]
                    msg = self.CheckCompletion(window_state)
                else:
                    msg = "Error: 'window_state' parameter is missing for CHECK_COMPLETION action."
            
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
    
        Obtain the array and k value in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the array and k value.
    
        Example Output:
            "{\"arr\": [1, 2, 1, 3, 4], \"k\": 2}"
        """
        return json.dumps({"arr": self.arr, "k": self.k})

    def InitializeSlidingWindow(self):
        r"""
    
        Initialize sliding window parameters, including left and right pointers, maximum length, unique count, and frequency map.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the initial sliding window state.
    
        Example Output:
            "{\"left\": 0, \"right\": 0, \"max_length\": 0, \"unique_count\": 0, \"freq_map\": {}}"
        """
        initial_state = {
            "left": 0,
            "right": 0,
            "max_length": 0,
            "unique_count": 0,
            "freq_map": {}
        }
        self.window_state = initial_state
        return json.dumps(initial_state)

    def MoveRightPointer(self, window_state):
        r"""
    
        Move the right pointer and update the frequency map and unique count.
    
        Args:
            window_state (dict): The current sliding window state, including left, right, max_length, unique_count, and freq_map.
    
        Returns:
            str: A JSON string of the updated sliding window state.
    
        Example Output:
            "{\"left\": 0, \"right\": 1, \"max_length\": 0, \"unique_count\": 1, \"freq_map\": {\"1\": 1}}"
        """
        n = len(self.arr)
        if window_state["right"] >= n:
            return json.dumps(window_state)
            
        # Convert freq_map keys back to integers
        freq_map = defaultdict(int, {int(k): v for k, v in window_state["freq_map"].items()})
        current_element = self.arr[window_state["right"]]
        
        if freq_map[current_element] == 0:
            window_state["unique_count"] += 1
        freq_map[current_element] += 1
        
        window_state["right"] += 1
        window_state["freq_map"] = dict(freq_map)  # Convert back to regular dict for JSON serialization
        
        return json.dumps(window_state)

    def AdjustLeftPointer(self, window_state):
        r"""
    
        When the unique count exceeds k, move the left pointer to shrink the window.
    
        Args:
            window_state (dict): The current sliding window state, including left, right, max_length, unique_count, and freq_map.
    
        Returns:
            str: A JSON string of the updated sliding window state.
    
        Example Output:
            "{\"left\": 1, \"right\": 3, \"max_length\": 0, \"unique_count\": 2, \"freq_map\": {\"1\": 1, \"2\": 1}}"
        """
        # Convert freq_map keys back to integers
        freq_map = defaultdict(int, {int(k): v for k, v in window_state["freq_map"].items()})
        
        while window_state["unique_count"] > self.k and window_state["left"] < window_state["right"]:
            left_element = self.arr[window_state["left"]]
            freq_map[left_element] -= 1
            
            if freq_map[left_element] == 0:
                window_state["unique_count"] -= 1
                
            window_state["left"] += 1
        
        window_state["freq_map"] = dict(freq_map)  # Convert back to regular dict for JSON serialization
        return json.dumps(window_state)

    def UpdateMaxLength(self, window_state):
        r"""
    
        When the unique count is equal to k, update the maximum length.
    
        Args:
            window_state (dict): The current sliding window state, including left, right, max_length, unique_count, and freq_map.
    
        Returns:
            str: A JSON string of the updated sliding window state.
    
        Example Output:
            "{\"left\": 0, \"right\": 3, \"max_length\": 3, \"unique_count\": 2, \"freq_map\": {\"1\": 2, \"2\": 1}}"
        """
        if window_state["unique_count"] == self.k:
            current_length = window_state["right"] - window_state["left"]
            if current_length > window_state["max_length"]:
                window_state["max_length"] = current_length
                
        return json.dumps(window_state)

    def CheckCompletion(self, window_state):
        r"""
    
        Check if the right pointer has reached the end of the array to determine if the traversal is completed.
    
        Args:
            window_state (dict): The current sliding window state, including left, right, max_length, unique_count, and freq_map.
    
        Returns:
            str: A JSON string indicating whether the traversal is completed and the current window state.
    
        Example Output:
            "{\"completed\": true, \"window_state\": {\"left\": 2, \"right\": 5, \"max_length\": 3, \"unique_count\": 2, \"freq_map\": {\"1\": 1, \"3\": 1, \"4\": 1}}}"
        """
        completed = window_state["right"] >= len(self.arr)
        return json.dumps({
            "completed": completed,
            "window_state": window_state
        })

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the maximum subarray length.
    
        Returns:
            str: Result information, including correctness and reward information.
    
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
        import json
        
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        observe_data = json.loads(observe_result)
        arr = observe_data['arr']
        k = observe_data['k']
        
        init_result = self.step(json.dumps({'name': 'InitializeSlidingWindow', 'parameters': {}}))[1]
        window_state = json.loads(init_result)
        
        while True:
            check_result = self.step(json.dumps({'name': 'CheckCompletion', 'parameters': {'window_state': window_state}}))[1]
            check_data = json.loads(check_result)
            if check_data['completed']:
                break
            
            move_right_result = self.step(json.dumps({'name': 'MoveRightPointer', 'parameters': {'window_state': window_state}}))[1]
            window_state = json.loads(move_right_result)
            
            adjust_left_result = self.step(json.dumps({'name': 'AdjustLeftPointer', 'parameters': {'window_state': window_state}}))[1]
            window_state = json.loads(adjust_left_result)
            
            update_max_result = self.step(json.dumps({'name': 'UpdateMaxLength', 'parameters': {'window_state': window_state}}))[1]
            window_state = json.loads(update_max_result)
        
        max_length = window_state['max_length']
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_arr1 = [1, 2, 1, 3, 4]
    test_k1 = 2
    env1 = MaxSubarrayWithKDistinctEnv.from_env_str(
        f"MaxSubarrayWithKDistinctEnv@{{\"arr\": {test_arr1}, \"k\": {test_k1}}}"
    )
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_arr2 = [3, 1, 2, 3, 3, 4, 5, 3, 2]
    test_k2 = 3
    env2 = MaxSubarrayWithKDistinctEnv.from_env_str(
        f"MaxSubarrayWithKDistinctEnv@{{\"arr\": {test_arr2}, \"k\": {test_k2}}}"
    )
    print(env2.solve())
    print("step count:", env2.step_count)