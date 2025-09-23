import gymnasium
import ast
import json

class ArithmeticSubarrayCountEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_DIFFERENCE = 0
        self.CHECK_SUBARRAY = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateDifference": self.CALCULATE_DIFFERENCE,
            "CheckSubarray": self.CHECK_SUBARRAY,
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
        prefix = "ArithmeticSubarrayCountEnv@"
        if not env_str.startswith(prefix):
            return None
        return ArithmeticSubarrayCountEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.array = options.get("array", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.array)
        if n < 3:
            return 0
        
        count = 0
        for start in range(n - 2):
            diff = self.array[start + 1] - self.array[start]
            for end in range(start + 2, n):
                valid = True
                for i in range(start + 1, end + 1):
                    if self.array[i] - self.array[i - 1] != diff:
                        valid = False
                        break
                if valid:
                    count += 1
                else:
                    break
        return count

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
            
            if action_code == self.CALCULATE_DIFFERENCE:
                if "index" in params:
                    index = params["index"]
                    msg = self.CalculateDifference(index)
                else:
                    msg = "Error: 'index' parameter is missing for CALCULATE_DIFFERENCE action."
            
            elif action_code == self.CHECK_SUBARRAY:
                if "start" in params and "end" in params and "diff" in params:
                    start = params["start"]
                    end = params["end"]
                    diff = params["diff"]
                    msg = self.CheckSubarray(start, end, diff)
                else:
                    msg = "Error: 'start', 'end' or 'diff' parameter is missing for CHECK_SUBARRAY action."
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
    def CalculateDifference(self, index: int):
        r"""
    
        Calculate the difference between the element at the specified index and the next element in the array.
    
        Args:
            index (int): The starting index for which to calculate the difference.
    
        Returns:
            str: The difference between the two elements.
    
        Example Output:
            "2"
        """
        if index < 0 or index >= len(self.array) - 1:
            return "Error: index out of range"
        diff = self.array[index + 1] - self.array[index]
        return str(diff)

    def CheckSubarray(self, start: int, end: int, diff: int):
        r"""
    
        Check if the subarray from start to end is an arithmetic subarray (with a common difference diff).
    
        Args:
            start (int): The starting index of the subarray.
            end (int): The ending index of the subarray.
            diff (int): The expected common difference.
    
        Returns:
            str: "True" indicates it is an arithmetic subarray, "False" indicates it is not.
    
        Example Output:
            "True"
        """
        if start < 0 or end >= len(self.array) or start >= end - 1:
            return "False"
            
        for i in range(start + 1, end + 1):
            if self.array[i] - self.array[i - 1] != diff:
                return "False"
        return "True"

    def Observe(self):
        r"""
    
        Return the array information in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string representation of the array.
    
        Example Output:
            "[2, 4, 6, 8]"
        """
        return str(self.array)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The number of arithmetic subarrays submitted by the user.
    
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
        array_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        array = ast.literal_eval(array_str)
        n = len(array)
        if n < 3:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        count = 0
        for start in range(n - 2):
            diff_str = self.step(json.dumps({'name': 'CalculateDifference', 'parameters': {'index': start}}))[1]
            diff = int(diff_str)
            for end in range(start + 2, n):
                check_result = self.step(json.dumps({'name': 'CheckSubarray', 'parameters': {'start': start, 'end': end, 'diff': diff}}))[1]
                if check_result == "True":
                    count += 1
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_array = [2, 4, 6, 8]
    env = ArithmeticSubarrayCountEnv.from_env_str(f"ArithmeticSubarrayCountEnv@{{\"array\": {test_array}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_array = [1, 3, 5, 7, 9]
    env = ArithmeticSubarrayCountEnv.from_env_str(f"ArithmeticSubarrayCountEnv@{{\"array\": {test_array}}}")
    print(env.solve())
    print("step count:", env.step_count)