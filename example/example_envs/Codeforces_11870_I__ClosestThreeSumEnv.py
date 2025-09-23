import gymnasium
import json
import ast
from typing import List

class ClosestThreeSumEnv(gymnasium.Env):
    def __init__(self, env_str: str = None):
        super().__init__()
        
        # [Required] Define the action names
        self.SORT_ARRAY = 0
        self.GET_ARRAY_LENGTH = 1
        self.CALCULATE_CURRENT_SUM = 2
        self.COMPARE_DISTANCE = 3
        self.MOVE_LEFT_POINTER = 4
        self.MOVE_RIGHT_POINTER = 5
        self.SET_CLOSEST_SUM = 6
        self.OBSERVE = 7
        self.DONE = 8
        
        # [Required] Define the action mapping
        self.func_mapping = {
            "SortArray": self.SORT_ARRAY,
            "GetArrayLength": self.GET_ARRAY_LENGTH,
            "CalculateCurrentSum": self.CALCULATE_CURRENT_SUM,
            "CompareDistance": self.COMPARE_DISTANCE,
            "MoveLeftPointer": self.MOVE_LEFT_POINTER,
            "MoveRightPointer": self.MOVE_RIGHT_POINTER,
            "SetClosestSum": self.SET_CLOSEST_SUM,
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
        prefix = "ClosestThreeSumEnv@"
        if not env_str.startswith(prefix):
            return None
        return ClosestThreeSumEnv(env_str=env_str)
    
    # [Required] Define the reset method of the environment
    def reset(self, options: dict = {}):
        self.arr = options.get("arr", [])
        self.target = options.get("target", 0)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."
    
    # [Required] Get the reference answer of the environment
    def get_ref_answer(self) -> int:
        arr_copy = sorted(self.arr)
        closest_sum = float('inf')
        
        for i in range(len(arr_copy) - 2):
            left, right = i + 1, len(arr_copy) - 1
            while left < right:
                current_sum = arr_copy[i] + arr_copy[left] + arr_copy[right]
                if abs(current_sum - self.target) < abs(closest_sum - self.target):
                    closest_sum = current_sum
                
                if current_sum < self.target:
                    left += 1
                elif current_sum > self.target:
                    right -= 1
                else:
                    return current_sum
        
        return closest_sum
    
    # [Required] Define the step method of the environment
    def step(self, action: str):
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
            
            if action_code == self.SORT_ARRAY:
                if "array" in params:
                    array = params["array"]
                    msg = self.SortArray(array)
                else:
                    msg = "Error: 'array' parameter is missing for SORT_ARRAY action."
            
            elif action_code == self.GET_ARRAY_LENGTH:
                if "array" in params:
                    array = params["array"]
                    msg = self.GetArrayLength(array)
                else:
                    msg = "Error: 'array' parameter is missing for GET_ARRAY_LENGTH action."
            
            elif action_code == self.CALCULATE_CURRENT_SUM:
                if "array" in params and "i" in params and "left" in params and "right" in params:
                    array = params["array"]
                    i = params["i"]
                    left = params["left"]
                    right = params["right"]
                    msg = self.CalculateCurrentSum(array, i, left, right)
                else:
                    msg = "Error: Parameters missing for CALCULATE_CURRENT_SUM action."
            
            elif action_code == self.COMPARE_DISTANCE:
                if "current_sum" in params and "closest_sum" in params and "target" in params:
                    current_sum = params["current_sum"]
                    closest_sum = params["closest_sum"]
                    target = params["target"]
                    msg = self.CompareDistance(current_sum, closest_sum, target)
                else:
                    msg = "Error: Parameters missing for COMPARE_DISTANCE action."
            
            elif action_code == self.MOVE_LEFT_POINTER:
                if "left" in params:
                    left = params["left"]
                    msg = self.MoveLeftPointer(left)
                else:
                    msg = "Error: 'left' parameter is missing for MOVE_LEFT_POINTER action."
            
            elif action_code == self.MOVE_RIGHT_POINTER:
                if "right" in params:
                    right = params["right"]
                    msg = self.MoveRightPointer(right)
                else:
                    msg = "Error: 'right' parameter is missing for MOVE_RIGHT_POINTER action."
            
            elif action_code == self.SET_CLOSEST_SUM:
                if "new_sum" in params:
                    new_sum = params["new_sum"]
                    msg = self.SetClosestSum(new_sum)
                else:
                    msg = "Error: 'new_sum' parameter is missing for SET_CLOSEST_SUM action."
            
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
    def SortArray(self, array: List[int]) -> str:
        r"""
    
        Sort the input array and return the sorted result.
        
        Args:
            array (List[int]): The integer array to be sorted.
            
        Returns:
            str: The sorted array returned as a JSON string.
            
        Example Output:
            "[-4, -1, 1, 2]"
        """
        sorted_array = sorted(array)
        return json.dumps(sorted_array)
    
    def GetArrayLength(self, array: List[int]) -> str:
        r"""
    
        Get the length of the input array.
        
        Args:
            array (List[int]): The array whose length is to be obtained.
            
        Returns:
            str: The length of the array returned as a string.
            
        Example Output:
            "4"
        """
        return str(len(array))
    
    def CalculateCurrentSum(self, array: List[int], i: int, left: int, right: int) -> str:
        r"""
    
        Calculate the sum of elements at three specified index positions in the array.
        
        Args:
            array (List[int]): The integer array.
            i (int): The index of the first element.
            left (int): The index of the second element.
            right (int): The index of the third element.
            
        Returns:
            str: The sum of the three elements returned as a string.
            
        Example Output:
            "2"
        """
        current_sum = array[i] + array[left] + array[right]
        return str(current_sum)
    
    def CompareDistance(self, current_sum: int, closest_sum: int, target: int) -> str:
        r"""
    
        Compare the distances of the current sum and the known closest sum to the target value, and return the sum that is closer.
        
        Args:
            current_sum (int): The currently calculated sum of three numbers.
            closest_sum (int): The known sum that is closest to the target value.
            target (int): The target value.
            
        Returns:
            str: The sum that is closer to the target value returned as a string. If the distances are equal, return the known closest sum.
            
        Example Output:
            "2"
        """
        current_distance = abs(current_sum - target)
        closest_distance = abs(closest_sum - target)
        
        if current_distance < closest_distance:
            return str(current_sum)
        return str(closest_sum)
    
    def MoveLeftPointer(self, left: int) -> str:
        r"""
    
        Move the left pointer one position to the right (increment by 1).
        
        Args:
            left (int): The current position of the left pointer.
            
        Returns:
            str: The position of the left pointer after moving returned as a string.
            
        Example Output:
            "2"
        """
        return str(left + 1)
    
    def MoveRightPointer(self, right: int) -> str:
        r"""
    
        Move the right pointer one position to the left (decrement by 1).
        
        Args:
            right (int): The current position of the right pointer.
            
        Returns:
            str: The position of the right pointer after moving returned as a string.
            
        Example Output:
            "2"
        """
        return str(right - 1)
    
    def SetClosestSum(self, new_sum: int) -> str:
        r"""
    
        Set a new closest sum and return a confirmation message.
        
        Args:
            new_sum (int): The new closest sum value.
            
        Returns:
            str: A confirmation message containing the newly set closest sum.
            
        Example Output:
            "Closest sum set to 2"
        """
        return f"Closest sum set to {new_sum}"
    
    def Observe(self) -> str:
        r"""
    
        Obtain information about the array and target value in the current environment.
        
        Args:
            None
            
        Returns:
            str: The array and target value in the current environment returned as a JSON string.
            
        Example Output:
            "{\"arr\": [-1, 2, 1, -4], \"target\": 1}"
        """
        return json.dumps({"arr": self.arr, "target": self.target})
    
    def Done(self, answer: int) -> str:
        r"""
    
        Submit the final answer and verify its correctness.
        
        Args:
            answer (int): The submitted answer of the sum of three numbers.
            
        Returns:
            str: Verification result information, including the submitted answer, reference answer, and reward value.
            
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        arr = observe_data['arr']
        target = observe_data['target']
        
        sorted_arr_str = self.step(json.dumps({'name': 'SortArray', 'parameters': {'array': arr}}))[1]
        sorted_arr = json.loads(sorted_arr_str)
        
        n_str = self.step(json.dumps({'name': 'GetArrayLength', 'parameters': {'array': sorted_arr}}))[1]
        n = int(n_str)
        
        initial_sum_str = self.step(json.dumps({
            'name': 'CalculateCurrentSum',
            'parameters': {'array': sorted_arr, 'i': 0, 'left': 1, 'right': 2}
        }))[1]
        initial_sum = int(initial_sum_str)
        self.step(json.dumps({'name': 'SetClosestSum', 'parameters': {'new_sum': initial_sum}}))
        closest_sum = initial_sum
        
        for i in range(n - 2):
            left = i + 1
            right = n - 1
            
            while left < right:
                current_sum_str = self.step(json.dumps({
                    'name': 'CalculateCurrentSum',
                    'parameters': {'array': sorted_arr, 'i': i, 'left': left, 'right': right}
                }))[1]
                current_sum = int(current_sum_str)
                
                new_closest_str = self.step(json.dumps({
                    'name': 'CompareDistance',
                    'parameters': {'current_sum': current_sum, 'closest_sum': closest_sum, 'target': target}
                }))[1]
                new_closest = int(new_closest_str)
                
                if new_closest != closest_sum:
                    closest_sum = new_closest
                    self.step(json.dumps({'name': 'SetClosestSum', 'parameters': {'new_sum': closest_sum}}))
                
                if current_sum < target:
                    left = int(self.step(json.dumps({
                        'name': 'MoveLeftPointer',
                        'parameters': {'left': left}
                    }))[1])
                else:
                    right = int(self.step(json.dumps({
                        'name': 'MoveRightPointer',
                        'parameters': {'right': right}
                    }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': closest_sum}}))[1]
# Test the environment
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    env1 = ClosestThreeSumEnv.from_env_str('ClosestThreeSumEnv@{"arr": [-1, 2, 1, -4], "target": 1}')
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # Test case 2
    print("\nTest Case 2:")
    env2 = ClosestThreeSumEnv.from_env_str('ClosestThreeSumEnv@{"arr": [1, 1, 1, 0], "target": -100}')
    print(env2.solve())
    print("step count:", env2.step_count)