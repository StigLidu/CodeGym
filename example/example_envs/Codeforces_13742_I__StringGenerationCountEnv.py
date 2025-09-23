import gymnasium
import ast
import json

class StringGenerationCountEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.PROCESS_CHARACTER = 0
        self.MULTIPLY_COUNTS = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "MultiplyCounts": self.MULTIPLY_COUNTS,
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
        prefix = "StringGenerationCountEnv@"
        if not env_str.startswith(prefix):
            return None
        return StringGenerationCountEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        count = 1
        for char in self.s:
            if char == '?':
                count *= 10
            elif char == '*':
                count *= 11  # 10 choices (0-9) plus 1 for empty sequence
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
            
            if action_code == self.PROCESS_CHARACTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.ProcessCharacter(char)
                else:
                    msg = "Error: 'char' parameter is missing for PROCESS_CHARACTER action."
            
            elif action_code == self.MULTIPLY_COUNTS:
                if "count1" in params and "count2" in params:
                    count1 = params["count1"]
                    count2 = params["count2"]
                    msg = self.MultiplyCounts(count1, count2)
                else:
                    msg = "Error: 'count1' or 'count2' parameter is missing for MULTIPLY_COUNTS action."
                    
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
    def ProcessCharacter(self, char: str):
        r"""
    
        Calculate the contribution value of a single character to the total number of generatable strings.
    
        Args:
            char (str): The character to be processed, which must be a digit, '?', or '*'.
    
        Returns:
            str: The contribution value of the character, where a digit is 1, '?' is 10, and '*' is 11.
    
        Example Output:
            "10"
        """
        if char == '?':
            return "10"
        elif char == '*':
            return "11"
        else:  # digit
            return "1"

    def MultiplyCounts(self, count1: int, count2: int):
        r"""
    
        Multiply two count values and return the product result.
    
        Args:
            count1 (int): The first count value.
            count2 (int): The second count value.
    
        Returns:
            str: The product of the two count values.
    
        Example Output:
            "100"
        """
        return str(count1 * count2)

    def Observe(self):
        r"""
    
        Obtain the special string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The special string in the current environment.
    
        Example Output:
            "1?2*34"
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the total number of different generatable strings.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 100, Reference answer: 100, Result: Correct, reward=1"
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
        s = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        total = 1
        for char in s:
            contribution = int(self.step(json.dumps({'name': 'ProcessCharacter', 'parameters': {'char': char}}))[1])
            total = int(self.step(json.dumps({'name': 'MultiplyCounts', 'parameters': {'count1': total, 'count2': contribution}}))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "1?2*34"
    env = StringGenerationCountEnv.from_env_str(f"StringGenerationCountEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "?*?"
    env = StringGenerationCountEnv.from_env_str(f"StringGenerationCountEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)