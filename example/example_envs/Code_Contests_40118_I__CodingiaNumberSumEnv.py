import gymnasium
import json
import ast

class CodingiaNumberSumEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.TRANSFORM_CHARACTER = 0
        self.CALCULATE_CHAR_VALUE = 1
        self.SUM_VALUES = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "TransformCharacter": self.TRANSFORM_CHARACTER,
            "CalculateCharValue": self.CALCULATE_CHAR_VALUE,
            "SumValues": self.SUM_VALUES,
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
        prefix = "CodingiaNumberSumEnv@"
        if not env_str.startswith(prefix):
            return None
        return CodingiaNumberSumEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.strings = options.get("strings", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        total_sum = 0
        for s in self.strings:
            for char in s:
                if char == 'z':
                    transformed_char = 'a'
                else:
                    transformed_char = chr(ord(char) + 1)
                total_sum += ord(transformed_char) - ord('a') + 1
        return total_sum

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
            
            if action_code == self.TRANSFORM_CHARACTER:
                if "character" in params:
                    character = params["character"]
                    msg = self.TransformCharacter(character)
                else:
                    msg = "Error: 'character' parameter is missing for TRANSFORM_CHARACTER action."
            
            elif action_code == self.CALCULATE_CHAR_VALUE:
                if "character" in params:
                    character = params["character"]
                    msg = self.CalculateCharValue(character)
                else:
                    msg = "Error: 'character' parameter is missing for CALCULATE_CHAR_VALUE action."
                    
            elif action_code == self.SUM_VALUES:
                if "values" in params:
                    values = params["values"]
                    msg = self.SumValues(values)
                else:
                    msg = "Error: 'values' parameter is missing for SUM_VALUES action."
                    
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
    def TransformCharacter(self, character: str):
        r"""
    
        Transform a single character according to the rules of Codingia. 'a' becomes 'b', 'b' becomes 'c', ..., 'z' becomes 'a'.
    
        Args:
            character (str): The single character to be transformed, which must be a lowercase letter.
    
        Returns:
            str: The transformed character.
    
        Example Output:
            "b"
        """
        if character == 'z':
            return 'a'
        else:
            return chr(ord(character) + 1)

    def CalculateCharValue(self, character: str):
        r"""
    
        Calculate the value of a character in the Codingia numbering system, where 'a' = 1, 'b' = 2, ..., 'z' = 26.
    
        Args:
            character (str): The single character whose value is to be calculated, which must be a lowercase letter.
    
        Returns:
            str: The value corresponding to the character.
    
        Example Output:
            "2"
        """
        return str(ord(character) - ord('a') + 1)

    def SumValues(self, values: list):
        r"""
    
        Calculate the sum of a list of values.
    
        Args:
            values (list[int]): The list of values for which the sum is to be calculated.
    
        Returns:
            str: The sum of the list of values.
    
        Example Output:
            "15"
        """
        return str(sum(values))

    def Observe(self):
        r"""
    
        Obtain the list of strings in the current environment.
    
        Args:
            None
    
        Returns:
            str: The list of strings in the current environment.
    
        Example Output:
            "[\"abc\", \"xyz\"]"
        """
        return json.dumps(self.strings)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 118, Reference answer: 118, Result: Correct, reward=1"
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
        str_list = json.loads(observe_result)
        
        all_values = []
        for s in str_list:
            for char in s:
                transformed_char = self.step(json.dumps({
                    'name': 'TransformCharacter',
                    'parameters': {'character': char}
                }))[1]
                char_value = int(self.step(json.dumps({
                    'name': 'CalculateCharValue',
                    'parameters': {'character': transformed_char}
                }))[1])
                all_values.append(char_value)
        
        total_sum = int(self.step(json.dumps({
            'name': 'SumValues',
            'parameters': {'values': all_values}
        }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total_sum}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - sample input
    print("Test Case 1:")
    test_strings1 = ["abc", "xyz", "hello"]
    env1 = CodingiaNumberSumEnv.from_env_str(f"CodingiaNumberSumEnv@{{\"strings\": {test_strings1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2 - random strings
    print("\nTest Case 2:")
    test_strings2 = ["a", "z", "abcz"]
    env2 = CodingiaNumberSumEnv.from_env_str(f"CodingiaNumberSumEnv@{{\"strings\": {test_strings2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)