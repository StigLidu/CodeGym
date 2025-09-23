import gymnasium
import ast
import json

class MinSubstringsEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.GET_STRING_LENGTH = 1
        self.GET_CHARACTER_AT_POSITION = 2
        self.COMPARE_ADJACENT_CHARACTERS = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "GetCharacterAtPosition": self.GET_CHARACTER_AT_POSITION,
            "CompareAdjacentCharacters": self.COMPARE_ADJACENT_CHARACTERS,
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
        prefix = "MinSubstringsEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinSubstringsEnv(env_str=env_str)

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
        if not self.s:
            return 0
        
        count = 1
        for i in range(1, len(self.s)):
            if self.s[i] != self.s[i-1]:
                count += 1
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
            
            if action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
            
            elif action_code == self.GET_CHARACTER_AT_POSITION:
                if "position" in params:
                    position = params["position"]
                    msg = self.GetCharacterAtPosition(position)
                else:
                    msg = "Error: 'position' parameter is missing for GET_CHARACTER_AT_POSITION action."
            
            elif action_code == self.COMPARE_ADJACENT_CHARACTERS:
                if "position" in params:
                    position = params["position"]
                    msg = self.CompareAdjacentCharacters(position)
                else:
                    msg = "Error: 'position' parameter is missing for COMPARE_ADJACENT_CHARACTERS action."
            
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
    def Observe(self):
        r"""
    
        Returns the string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string in the current environment.
    
        Example Output:
            "LLRRPLP"
        """
        return self.s

    def GetStringLength(self):
        r"""
    
        Gets the length of the current string.
    
        Args:
            None
    
        Returns:
            str: The length of the string.
    
        Example Output:
            "7"
        """
        return str(len(self.s))

    def GetCharacterAtPosition(self, position: int):
        r"""
    
        Gets the character at the specified position in the string.
    
        Args:
            position (int): The position index of the character to be retrieved (starting from 0).
    
        Returns:
            str: The character at the specified position.
    
        Example Output:
            "L"
        """
        if 0 <= position < len(self.s):
            return self.s[position]
        else:
            return "Error: position out of range"

    def CompareAdjacentCharacters(self, position: int):
        r"""
    
        Compares whether the character at the specified position in the string is the same as the previous position.
    
        Args:
            position (int): The current position index (starting from 1).
    
        Returns:
            str: "True" if they are the same, "False" if they are different, and an error message if there is an error.
    
        Example Output:
            "False"
        """
        if 1 <= position < len(self.s):
            return str(self.s[position] == self.s[position-1])
        else:
            return "Error: position out of range"

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 5, Reference answer: 5, Result: Correct, reward=1"
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
        length_str = self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1]
        length = int(length_str)
        
        if length == 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        count = 1
        
        for position in range(1, length):
            compare_result = self.step(json.dumps({'name': 'CompareAdjacentCharacters', 'parameters': {'position': position}}))[1]
            if compare_result == "False":
                count += 1
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "LLRRPLP"
    env = MinSubstringsEnv.from_env_str(f"MinSubstringsEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "LRLRLR"
    env = MinSubstringsEnv.from_env_str(f"MinSubstringsEnv@{{\"s\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)