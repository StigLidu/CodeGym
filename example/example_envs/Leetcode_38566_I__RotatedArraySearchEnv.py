# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class RotatedArraySearchEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_POINTERS = 0
        self.CALCULATE_MID = 1
        self.CHECK_MID_VALUE = 2
        self.CHECK_LEFT_SORTED = 3
        self.CHECK_TARGET_IN_LEFT_RANGE = 4
        self.CHECK_TARGET_IN_RIGHT_RANGE = 5
        self.MOVE_LEFT_POINTER = 6
        self.MOVE_RIGHT_POINTER = 7
        self.OBSERVE = 8
        self.DONE = 9

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializePointers": self.INITIALIZE_POINTERS,
            "CalculateMid": self.CALCULATE_MID,
            "CheckMidValue": self.CHECK_MID_VALUE,
            "CheckLeftSorted": self.CHECK_LEFT_SORTED,
            "CheckTargetInLeftRange": self.CHECK_TARGET_IN_LEFT_RANGE,
            "CheckTargetInRightRange": self.CHECK_TARGET_IN_RIGHT_RANGE,
            "MoveLeftPointer": self.MOVE_LEFT_POINTER,
            "MoveRightPointer": self.MOVE_RIGHT_POINTER,
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
        prefix = "RotatedArraySearchEnv@"
        if not env_str.startswith(prefix):
            return None
        return RotatedArraySearchEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.nums = options.get("nums", [])
        self.target = options.get("target", 0)
        self.left = None
        self.right = None
        self.mid = None
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.nums:
            return -1
        
        left, right = 0, len(self.nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2

            if self.nums[mid] == self.target:
                return mid
            
            # Check if the left side is sorted
            if self.nums[left] <= self.nums[mid]:
                if self.nums[left] <= self.target < self.nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Otherwise, the right side must be sorted
            else:
                if self.nums[mid] < self.target <= self.nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1

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
            
            if action_code == self.INITIALIZE_POINTERS:
                msg = self.InitializePointers()
            
            elif action_code == self.CALCULATE_MID:
                if "left" in params and "right" in params:
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateMid(left, right)
                else:
                    msg = "Error: 'left' or 'right' parameter is missing for CALCULATE_MID action."
            
            elif action_code == self.CHECK_MID_VALUE:
                if "mid" in params:
                    mid = params["mid"]
                    msg = self.CheckMidValue(mid)
                else:
                    msg = "Error: 'mid' parameter is missing for CHECK_MID_VALUE action."
            
            elif action_code == self.CHECK_LEFT_SORTED:
                if "left" in params and "mid" in params:
                    left = params["left"]
                    mid = params["mid"]
                    msg = self.CheckLeftSorted(left, mid)
                else:
                    msg = "Error: 'left' or 'mid' parameter is missing for CHECK_LEFT_SORTED action."
            
            elif action_code == self.CHECK_TARGET_IN_LEFT_RANGE:
                if "left" in params and "mid" in params and "target" in params:
                    left = params["left"]
                    mid = params["mid"]
                    target = params["target"]
                    msg = self.CheckTargetInLeftRange(left, mid, target)
                else:
                    msg = "Error: 'left', 'mid' or 'target' parameter is missing for CHECK_TARGET_IN_LEFT_RANGE action."
            
            elif action_code == self.CHECK_TARGET_IN_RIGHT_RANGE:
                if "mid" in params and "right" in params and "target" in params:
                    mid = params["mid"]
                    right = params["right"]
                    target = params["target"]
                    msg = self.CheckTargetInRightRange(mid, right, target)
                else:
                    msg = "Error: 'mid', 'right' or 'target' parameter is missing for CHECK_TARGET_IN_RIGHT_RANGE action."
            
            elif action_code == self.MOVE_LEFT_POINTER:
                if "position" in params:
                    position = params["position"]
                    msg = self.MoveLeftPointer(position)
                else:
                    msg = "Error: 'position' parameter is missing for MOVE_LEFT_POINTER action."
            
            elif action_code == self.MOVE_RIGHT_POINTER:
                if "position" in params:
                    position = params["position"]
                    msg = self.MoveRightPointer(position)
                else:
                    msg = "Error: 'position' parameter is missing for MOVE_RIGHT_POINTER action."
            
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
    def InitializePointers(self):
        r"""
    
        Initialize the left and right pointers, set the left pointer to 0, and the right pointer to the length of the array minus 1.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the initialized values of the left and right pointers.
    
        Example Output:
            "{\"left\": 0, \"right\": 6}"
        """
        self.left = 0
        self.right = len(self.nums) - 1 if self.nums else 0
        return json.dumps({"left": self.left, "right": self.right})

    def CalculateMid(self, left: int, right: int):
        r"""
    
        Calculate the middle position based on the left and right pointers.
    
        Args:
            left (int): Position of the left pointer
            right (int): Position of the right pointer
    
        Returns:
            str: A JSON string containing the calculated middle position.
    
        Example Output:
            "{\"mid\": 3}"
        """
        self.mid = left + (right - left) // 2
        return json.dumps({"mid": self.mid})

    def CheckMidValue(self, mid: int):
        r"""
    
        Check if the element at the middle position is equal to the target value.
    
        Args:
            mid (int): Index of the middle position
    
        Returns:
            str: A JSON string containing the check result, where 'found' is True if the target value is found, and 'index' is the index of the target value.
    
        Example Output:
            "{\"found\": true, \"index\": 4}"
        """
        if mid < 0 or mid >= len(self.nums):
            return json.dumps({"found": False, "index": -1})
        
        found = self.nums[mid] == self.target
        return json.dumps({"found": found, "index": mid if found else -1})

    def CheckLeftSorted(self, left: int, mid: int):
        r"""
    
        Check if the left half of the array is sorted.
    
        Args:
            left (int): Position of the left pointer
            mid (int): Index of the middle position
    
        Returns:
            str: A JSON string containing the check result, where 'is_sorted' is True if the left half is sorted.
    
        Example Output:
            "{\"is_sorted\": true}"
        """
        if left < 0 or mid >= len(self.nums):
            return json.dumps({"is_sorted": False})
            
        is_sorted = self.nums[left] <= self.nums[mid]
        return json.dumps({"is_sorted": is_sorted})

    def CheckTargetInLeftRange(self, left: int, mid: int, target: int):
        r"""
    
        Check if the target value is within the range of the left half of the array.
    
        Args:
            left (int): Position of the left pointer
            mid (int): Index of the middle position
            target (int): Target value
    
        Returns:
            str: A JSON string containing the check result, where 'in_range' is True if the target value is within the left half range.
    
        Example Output:
            "{\"in_range\": true}"
        """
        if left < 0 or mid >= len(self.nums):
            return json.dumps({"in_range": False})
            
        in_range = self.nums[left] <= target < self.nums[mid]
        return json.dumps({"in_range": in_range})

    def CheckTargetInRightRange(self, mid: int, right: int, target: int):
        r"""
    
        Check if the target value is within the range of the right half of the array.
    
        Args:
            mid (int): Index of the middle position
            right (int): Position of the right pointer
            target (int): Target value
    
        Returns:
            str: A JSON string containing the check result, where 'in_range' is True if the target value is within the right half range.
    
        Example Output:
            "{\"in_range\": false}"
        """
        if mid < 0 or right >= len(self.nums):
            return json.dumps({"in_range": False})
            
        in_range = self.nums[mid] < target <= self.nums[right]
        return json.dumps({"in_range": in_range})

    def MoveLeftPointer(self, position: int):
        r"""
    
        Move the left pointer to the specified position.
    
        Args:
            position (int): New position of the left pointer
    
        Returns:
            str: A JSON string containing the value of the left pointer after moving.
    
        Example Output:
            "{\"left\": 4}"
        """
        self.left = position
        return json.dumps({"left": self.left})

    def MoveRightPointer(self, position: int):
        r"""
    
        Move the right pointer to the specified position.
    
        Args:
            position (int): New position of the right pointer
    
        Returns:
            str: A JSON string containing the value of the right pointer after moving.
    
        Example Output:
            "{\"right\": 2}"
        """
        self.right = position
        return json.dumps({"right": self.right})

    def Observe(self):
        r"""
    
        Obtain the current environment state, including the array, target value, and pointer positions.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the current environment state.
    
        Example Output:
            "{\"nums\": [4,5,6,7,0,1,2], \"target\": 0, \"left\": 0, \"right\": 6, \"mid\": null}"
        """
        state = {
            "nums": self.nums,
            "target": self.target,
            "left": self.left,
            "right": self.right,
            "mid": self.mid
        }
        return json.dumps(state)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, which is the index of the target value or -1.
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: 4, Reference answer: 4, Result: Correct, reward=1"
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
        
        init_result = self.step(json.dumps({"name": "InitializePointers", "parameters": {}}))[1]
        init_data = json.loads(init_result)
        left = init_data["left"]
        right = init_data["right"]
        
        observe_result = self.step(json.dumps({"name": "Observe", "parameters": {}}))[1]
        observe_data = json.loads(observe_result)
        target = observe_data["target"]
        
        while left <= right:
            mid_result = self.step(json.dumps({
                "name": "CalculateMid",
                "parameters": {"left": left, "right": right}
            }))[1]
            mid = json.loads(mid_result)["mid"]
            
            check_mid = self.step(json.dumps({
                "name": "CheckMidValue",
                "parameters": {"mid": mid}
            }))[1]
            check_mid_data = json.loads(check_mid)
            if check_mid_data["found"]:
                return self.step(json.dumps({
                    "name": "Done",
                    "parameters": {"answer": check_mid_data["index"]}
                }))[1]
            
            left_sorted = self.step(json.dumps({
                "name": "CheckLeftSorted",
                "parameters": {"left": left, "mid": mid}
            }))[1]
            left_sorted_data = json.loads(left_sorted)
            
            if left_sorted_data["is_sorted"]:
                target_left = self.step(json.dumps({
                    "name": "CheckTargetInLeftRange",
                    "parameters": {"left": left, "mid": mid, "target": target}
                }))[1]
                target_left_data = json.loads(target_left)
                
                if target_left_data["in_range"]:
                    right = mid - 1
                    self.step(json.dumps({
                        "name": "MoveRightPointer",
                        "parameters": {"position": right}
                    }))
                else:
                    left = mid + 1
                    self.step(json.dumps({
                        "name": "MoveLeftPointer",
                        "parameters": {"position": left}
                    }))
            else:
                target_right = self.step(json.dumps({
                    "name": "CheckTargetInRightRange",
                    "parameters": {"mid": mid, "right": right, "target": target}
                }))[1]
                target_right_data = json.loads(target_right)
                
                if target_right_data["in_range"]:
                    left = mid + 1
                    self.step(json.dumps({
                        "name": "MoveLeftPointer",
                        "parameters": {"position": left}
                    }))
                else:
                    right = mid - 1
                    self.step(json.dumps({
                        "name": "MoveRightPointer",
                        "parameters": {"position": right}
                    }))
        
        return self.step(json.dumps({
            "name": "Done",
            "parameters": {"answer": -1}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_nums = [4,5,6,7,0,1,2]
    test_target = 0
    env = RotatedArraySearchEnv.from_env_str(f"RotatedArraySearchEnv@{{\"nums\": {test_nums}, \"target\": {test_target}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_nums = [4,5,6,7,0,1,2]
    test_target = 3
    env = RotatedArraySearchEnv.from_env_str(f"RotatedArraySearchEnv@{{\"nums\": {test_nums}, \"target\": {test_target}}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_nums = [1]
    test_target = 1
    env = RotatedArraySearchEnv.from_env_str(f"RotatedArraySearchEnv@{{\"nums\": {test_nums}, \"target\": {test_target}}}")
    print(env.solve())
    print("step count:", env.step_count)