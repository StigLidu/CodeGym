# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class MinRoomsEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.GET_SORTED_START_TIMES = 0
        self.GET_SORTED_END_TIMES = 1
        self.COMPARE_TIMES = 2
        self.UPDATE_ROOMS_COUNT = 3
        self.UPDATE_MAX_ROOMS = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "GetSortedStartTimes": self.GET_SORTED_START_TIMES,
            "GetSortedEndTimes": self.GET_SORTED_END_TIMES,
            "CompareTimes": self.COMPARE_TIMES,
            "UpdateRoomsCount": self.UPDATE_ROOMS_COUNT,
            "UpdateMaxRooms": self.UPDATE_MAX_ROOMS,
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
        prefix = "MinRoomsEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinRoomsEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.tasks = options.get("tasks", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.tasks:
            return 0

        start_times = sorted([task[0] for task in self.tasks])
        end_times = sorted([task[1] for task in self.tasks])
        
        rooms_needed = 0
        max_rooms = 0
        start_ptr, end_ptr = 0, 0
        n = len(self.tasks)
        
        while start_ptr < n:
            if start_times[start_ptr] < end_times[end_ptr]:
                rooms_needed += 1
                max_rooms = max(max_rooms, rooms_needed)
                start_ptr += 1
            else:
                rooms_needed -= 1
                end_ptr += 1
        
        return max_rooms

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
            
            if action_code == self.GET_SORTED_START_TIMES:
                msg = self.GetSortedStartTimes()
            
            elif action_code == self.GET_SORTED_END_TIMES:
                msg = self.GetSortedEndTimes()
                
            elif action_code == self.COMPARE_TIMES:
                if "start_idx" in params and "end_idx" in params and "start_times" in params and "end_times" in params:
                    start_idx = params["start_idx"]
                    end_idx = params["end_idx"]
                    start_times = params["start_times"]
                    end_times = params["end_times"]
                    msg = self.CompareTimes(start_idx, end_idx, start_times, end_times)
                else:
                    msg = "Error: Parameters missing for COMPARE_TIMES action."
                    
            elif action_code == self.UPDATE_ROOMS_COUNT:
                if "current_count" in params and "operation" in params:
                    current_count = params["current_count"]
                    operation = params["operation"]
                    msg = self.UpdateRoomsCount(current_count, operation)
                else:
                    msg = "Error: Parameters missing for UPDATE_ROOMS_COUNT action."
                    
            elif action_code == self.UPDATE_MAX_ROOMS:
                if "current_max" in params and "current_count" in params:
                    current_max = params["current_max"]
                    current_count = params["current_count"]
                    msg = self.UpdateMaxRooms(current_max, current_count)
                else:
                    msg = "Error: Parameters missing for UPDATE_MAX_ROOMS action."
                    
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
    def GetSortedStartTimes(self):
        r"""
    
        Retrieve the start times of all tasks and sort them in ascending order.
    
        Args:
            None
    
        Returns:
            str: A string representation of the sorted list of start times.
    
        Example Output:
            "[0, 5, 15]"
        """
        start_times = sorted([task[0] for task in self.tasks])
        return json.dumps(start_times)

    def GetSortedEndTimes(self):
        r"""
    
        Retrieve the end times of all tasks and sort them in ascending order.
    
        Args:
            None
    
        Returns:
            str: A string representation of the sorted list of end times.
    
        Example Output:
            "[10, 20, 30]"
        """
        end_times = sorted([task[1] for task in self.tasks])
        return json.dumps(end_times)

    def CompareTimes(self, start_idx: int, end_idx: int, start_times: list, end_times: list):
        r"""
    
        Compare the start time and end time at the specified indices.
    
        Args:
            start_idx (int): The index of the start time to be compared.
            end_idx (int): The index of the end time to be compared.
            start_times (list): The sorted list of start times.
            end_times (list): The sorted list of end times.
    
        Returns:
            str: "True" indicates that the start time is less than the end time, "False" indicates that the start time is greater than or equal to the end time.
    
        Example Output:
            "True"
        """
        result = start_times[start_idx] < end_times[end_idx]
        return str(result)

    def UpdateRoomsCount(self, current_count: int, operation: str):
        r"""
    
        Update the current number of required meeting rooms.
    
        Args:
            current_count (int): The current number of meeting rooms.
            operation (str): The type of operation; "add" means increment by 1, "subtract" means decrement by 1.
    
        Returns:
            str: The updated number of meeting rooms.
    
        Example Output:
            "2"
        """
        if operation == "add":
            new_count = current_count + 1
        elif operation == "subtract":
            new_count = current_count - 1
        else:
            return f"Error: Invalid operation {operation}"
        return str(new_count)

    def UpdateMaxRooms(self, current_max: int, current_count: int):
        r"""
    
        Update the maximum number of required meeting rooms.
    
        Args:
            current_max (int): The current maximum number of meeting rooms.
            current_count (int): The current number of required meeting rooms.
    
        Returns:
            str: The updated maximum number of meeting rooms.
    
        Example Output:
            "3"
        """
        new_max = max(current_max, current_count)
        return str(new_max)

    def Observe(self):
        r"""
    
        Retrieve the list of tasks in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current list of tasks.
    
        Example Output:
            "[[0, 30], [5, 10], [15, 20]]"
        """
        return json.dumps(self.tasks)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The minimum number of meeting rooms submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        import ast
        start_times_str = self.step(json.dumps({'name': 'GetSortedStartTimes', 'parameters': {}}))[1]
        start_times = ast.literal_eval(start_times_str)
        end_times_str = self.step(json.dumps({'name': 'GetSortedEndTimes', 'parameters': {}}))[1]
        end_times = ast.literal_eval(end_times_str)
        
        start_idx = 0
        end_idx = 0
        current_count = 0
        current_max = 0
        
        while start_idx < len(start_times) and end_idx < len(end_times):
            compare_result = self.step(json.dumps({
                'name': 'CompareTimes',
                'parameters': {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_times': start_times,
                    'end_times': end_times
                }
            }))[1]
            
            if compare_result == "True":
                current_count = int(self.step(json.dumps({
                    'name': 'UpdateRoomsCount',
                    'parameters': {
                        'current_count': current_count,
                        'operation': 'add'
                    }
                }))[1])
                current_max = int(self.step(json.dumps({
                    'name': 'UpdateMaxRooms',
                    'parameters': {
                        'current_max': current_max,
                        'current_count': current_count
                    }
                }))[1])
                start_idx += 1
            else:
                current_count = int(self.step(json.dumps({
                    'name': 'UpdateRoomsCount',
                    'parameters': {
                        'current_count': current_count,
                        'operation': 'subtract'
                    }
                }))[1])
                end_idx += 1
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': current_max}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_tasks1 = [[0, 30], [5, 10], [15, 20]]
    env1 = MinRoomsEnv.from_env_str(f"MinRoomsEnv@{{\"tasks\": {test_tasks1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_tasks2 = [[1, 4], [2, 5], [7, 9], [6, 8]]
    env2 = MinRoomsEnv.from_env_str(f"MinRoomsEnv@{{\"tasks\": {test_tasks2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)