import gymnasium
import json
import ast

class PartitionCheckerEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_TOTAL_SUM = 0
        self.CHECK_IF_EVEN = 1
        self.CALCULATE_TARGET = 2
        self.UPDATE_SUBSET_SUMS = 3
        self.CHECK_TARGET_EXISTS = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateTotalSum": self.CALCULATE_TOTAL_SUM,
            "CheckIfEven": self.CHECK_IF_EVEN,
            "CalculateTarget": self.CALCULATE_TARGET,
            "UpdateSubsetSums": self.UPDATE_SUBSET_SUMS,
            "CheckTargetExists": self.CHECK_TARGET_EXISTS,
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
        prefix = "PartitionCheckerEnv@"
        if not env_str.startswith(prefix):
            return None
        return PartitionCheckerEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.nums = options.get("nums", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        total_sum = sum(self.nums)
        if total_sum % 2 != 0:
            return False

        target = total_sum // 2
        subset_sum = set([0])

        for num in self.nums:
            new_sums = set()
            for s in subset_sum:
                if s + num == target:
                    return True
                elif s + num < target:
                    new_sums.add(s + num)
            subset_sum.update(new_sums)
        
        return target in subset_sum

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
            
            if action_code == self.CALCULATE_TOTAL_SUM:
                msg = self.CalculateTotalSum()
            
            elif action_code == self.CHECK_IF_EVEN:
                if "total_sum" in params:
                    total_sum = params["total_sum"]
                    msg = self.CheckIfEven(total_sum)
                else:
                    msg = "Error: 'total_sum' parameter is missing for CHECK_IF_EVEN action."
            
            elif action_code == self.CALCULATE_TARGET:
                if "total_sum" in params:
                    total_sum = params["total_sum"]
                    msg = self.CalculateTarget(total_sum)
                else:
                    msg = "Error: 'total_sum' parameter is missing for CALCULATE_TARGET action."
            
            elif action_code == self.UPDATE_SUBSET_SUMS:
                if "subset_sums" in params and "num" in params and "target" in params:
                    subset_sums = set(params["subset_sums"])
                    num = params["num"]
                    target = params["target"]
                    result = self.UpdateSubsetSums(subset_sums, num, target)
                    msg = json.dumps(result)
                else:
                    msg = "Error: 'subset_sums', 'num' or 'target' parameter is missing for UPDATE_SUBSET_SUMS action."
            
            elif action_code == self.CHECK_TARGET_EXISTS:
                if "subset_sums" in params and "target" in params:
                    subset_sums = set(params["subset_sums"])
                    target = params["target"]
                    msg = self.CheckTargetExists(subset_sums, target)
                else:
                    msg = "Error: 'subset_sums' or 'target' parameter is missing for CHECK_TARGET_EXISTS action."
            
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
    def CalculateTotalSum(self):
        r"""
    
        Calculate the sum of all elements in the current list.
    
        Args:
            None
    
        Returns:
            str: The sum of the list elements.
    
        Example Output:
            "22"
        """
        total_sum = sum(self.nums)
        return str(total_sum)

    def CheckIfEven(self, total_sum: int):
        r"""
    
        Check if the total sum is an even number.
    
        Args:
            total_sum (int): The sum of the list elements.
    
        Returns:
            str: Returns "true" if it is an even number, otherwise returns "false".
    
        Example Output:
            "true"
        """
        return "true" if total_sum % 2 == 0 else "false"

    def CalculateTarget(self, total_sum: int):
        r"""
    
        Calculate the target value, which is half of the total sum.
    
        Args:
            total_sum (int): The sum of the list elements.
    
        Returns:
            str: The target value.
    
        Example Output:
            "11"
        """
        target = total_sum // 2
        return str(target)

    def UpdateSubsetSums(self, subset_sums: set, num: int, target: int):
        r"""
    
        Update the set of possible subset sums and check if the target value is reached.
    
        Args:
            subset_sums (set[int]): The current set of possible subset sums.
            num (int): The current number to be processed.
            target (int): The target subset sum.
    
        Returns:
            str: A JSON string of a dictionary containing the updated set of subset sums and whether the target is reached.
                The dictionary includes the key "subset_sums" (with the value being the list of updated subset sums) and the key "found_target" (with the value being a boolean).
    
        Example Output:
            "{\"subset_sums\": [0, 1, 5, 6], \"found_target\": false}"
        """
        new_sums = set()
        found_target = False
        
        for s in subset_sums:
            if s + num == target:
                found_target = True
                break
            elif s + num < target:
                new_sums.add(s + num)
        
        if not found_target:
            subset_sums.update(new_sums)
            
        return {
            "subset_sums": list(subset_sums),
            "found_target": found_target
        }

    def CheckTargetExists(self, subset_sums: set, target: int):
        r"""
    
        Check if the target value is in the set of subset sums.
    
        Args:
            subset_sums (set[int]): The set of possible subset sums.
            target (int): The target subset sum.
    
        Returns:
            str: Returns "true" if the target value exists, otherwise returns "false".
    
        Example Output:
            "true"
        """
        return "true" if target in subset_sums else "false"

    def Observe(self):
        r"""
    
        Return the current list of integers.
    
        Args:
            None
    
        Returns:
            str: The current list of integers.
    
        Example Output:
            "[1, 5, 11, 5]"
        """
        return str(self.nums)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (bool): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: true, Reference answer: true, Result: Correct, reward=1"
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
        nums_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        nums = ast.literal_eval(nums_str)
        
        total_sum_str = self.step(json.dumps({'name': 'CalculateTotalSum', 'parameters': {}}))[1]
        total_sum = int(total_sum_str)
        
        is_even_str = self.step(json.dumps({'name': 'CheckIfEven', 'parameters': {'total_sum': total_sum}}))[1]
        if is_even_str == "false":
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': False}}))[1]
        
        target_str = self.step(json.dumps({'name': 'CalculateTarget', 'parameters': {'total_sum': total_sum}}))[1]
        target = int(target_str)
        
        subset_sums = {0}
        found = False
        
        for num in nums:
            update_result_str = self.step(json.dumps({
                'name': 'UpdateSubsetSums',
                'parameters': {
                    'subset_sums': list(subset_sums),
                    'num': num,
                    'target': target
                }
            }))[1]
            update_result = json.loads(update_result_str)
            subset_sums = set(update_result['subset_sums'])
            if update_result['found_target']:
                found = True
                break
        
        if not found:
            check_str = self.step(json.dumps({
                'name': 'CheckTargetExists',
                'parameters': {
                    'subset_sums': list(subset_sums),
                    'target': target
                }
            }))[1]
            found = (check_str == "true")
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': found}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_nums1 = [1, 5, 11, 5]
    env1 = PartitionCheckerEnv.from_env_str(f"PartitionCheckerEnv@{{\"nums\": {test_nums1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_nums2 = [1, 2, 3, 5]
    env2 = PartitionCheckerEnv.from_env_str(f"PartitionCheckerEnv@{{\"nums\": {test_nums2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)