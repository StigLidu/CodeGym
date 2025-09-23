import gymnasium
import ast
import json

class MagicalPrefixEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_CHARACTER_AT_POSITION = 0
        self.GET_NEXT_EXPECTED_CHARACTER = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckCharacterAtPosition": self.CHECK_CHARACTER_AT_POSITION,
            "GetNextExpectedCharacter": self.GET_NEXT_EXPECTED_CHARACTER,
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
        prefix = "MagicalPrefixEnv@"
        if not env_str.startswith(prefix):
            return None
        return MagicalPrefixEnv(env_str=env_str)

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
        expected_char = 'a'
        length = 0

        for char in self.s:
            if char == expected_char:
                length += 1
                expected_char = chr(ord(expected_char) + 1)
            else:
                break

        return length

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
            
            if action_code == self.CHECK_CHARACTER_AT_POSITION:
                if "position" in params and "expected_char" in params:
                    position = params["position"]
                    expected_char = params["expected_char"]
                    msg = self.CheckCharacterAtPosition(position, expected_char)
                else:
                    msg = "Error: 'position' or 'expected_char' parameter is missing for CHECK_CHARACTER_AT_POSITION action."
            
            elif action_code == self.GET_NEXT_EXPECTED_CHARACTER:
                if "current_char" in params:
                    current_char = params["current_char"]
                    msg = self.GetNextExpectedCharacter(current_char)
                else:
                    msg = "Error: 'current_char' parameter is missing for GET_NEXT_EXPECTED_CHARACTER action."
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
    def CheckCharacterAtPosition(self, position: int, expected_char: str):
        r"""
    
        Check if the character at the specified position in the string is the same as the expected character.
    
        Args:
            position (int): The position to check (counting starts from 0).
            expected_char (str): The expected character.
    
        Returns:
            str: Returns "True" if the character at the specified position is the same as the expected character, otherwise returns "False"; also returns "False" if the position exceeds the length of the string.
    
        Example Output:
            "True"
        """
        if position >= len(self.s):
            return "False"
        return "True" if self.s[position] == expected_char else "False"

    def GetNextExpectedCharacter(self, current_char: str):
        r"""
    
        Get the next character in the alphabet after the current character.
    
        Args:
            current_char (str): The current character.
    
        Returns:
            str: The next letter after the current character; returns '{' (a character beyond the alphabet range) if the current character is 'z'.
    
        Example Output:
            "b"
        """
        return chr(ord(current_char) + 1)

    def Observe(self):
        r"""
    
        Return the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "abcxyza"
        """
        return self.s

    def Done(self, answer: int):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The length of the longest magic prefix submitted by the user.
    
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
        self.step(json.dumps({'name': 'Observe', 'parameters': {}}))
        
        max_length = 0
        current_expected_char = 'a'
        current_position = 0
        
        while True:
            check_result = self.step(json.dumps({
                'name': 'CheckCharacterAtPosition',
                'parameters': {'position': current_position, 'expected_char': current_expected_char}
            }))[1]
            
            if check_result == "True":
                max_length += 1
                current_expected_char = self.step(json.dumps({
                    'name': 'GetNextExpectedCharacter',
                    'parameters': {'current_char': current_expected_char}
                }))[1]
                current_position += 1
            else:
                break
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_length}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "abcxyza"
    env = MagicalPrefixEnv.from_env_str(f"MagicalPrefixEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "abdc"
    env = MagicalPrefixEnv.from_env_str(f"MagicalPrefixEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_string = "a"
    env = MagicalPrefixEnv.from_env_str(f"MagicalPrefixEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)