import gymnasium
import ast
import json

class RemoveDuplicatesEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_CHARACTER = 1
        self.ADD_CHARACTER = 2
        self.GET_CURRENT_RESULT = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckCharacter": self.CHECK_CHARACTER,
            "AddCharacter": self.ADD_CHARACTER,
            "GetCurrentResult": self.GET_CURRENT_RESULT,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property of the environment
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "RemoveDuplicatesEnv@"
        if not env_str.startswith(prefix):
            return None
        return RemoveDuplicatesEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.input_string = options.get("input_string", "")
        self.seen_chars = set()
        self.result_chars = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        seen = set()
        result = []
        for char in self.input_string:
            if char not in seen:
                seen.add(char)
                result.append(char)
        return ''.join(result)

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
            
            elif action_code == self.CHECK_CHARACTER:
                if "character" in params:
                    character = params["character"]
                    msg = self.CheckCharacter(character)
                else:
                    msg = "Error: 'character' parameter is missing for CHECK_CHARACTER action."
            
            elif action_code == self.ADD_CHARACTER:
                if "character" in params:
                    character = params["character"]
                    msg = self.AddCharacter(character)
                else:
                    msg = "Error: 'character' parameter is missing for ADD_CHARACTER action."
                    
            elif action_code == self.GET_CURRENT_RESULT:
                msg = self.GetCurrentResult()
                
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
    
        Returns the current input string that needs to be processed.
    
        Args:
            None
    
        Returns:
            str: The current input string to be processed.
    
        Example Output:
            "Programming"
        """
        return self.input_string

    def CheckCharacter(self, character: str):
        r"""
    
        Checks whether the character has been recorded.
    
        Args:
            character (str): The single character to be checked.
    
        Returns:
            str: "True" indicates the character has been recorded, "False" indicates it has not been recorded.
    
        Example Output:
            "False"
        """
        return str(character in self.seen_chars)

    def AddCharacter(self, character: str):
        r"""
    
        Adds the character to the result list and marks it as recorded.
    
        Args:
            character (str): The single character to be added.
    
        Returns:
            str: The string representation of the result list after addition.
    
        Example Output:
            "['P', 'r', 'o']"
        """
        if character not in self.seen_chars:
            self.seen_chars.add(character)
            self.result_chars.append(character)
        return str(self.result_chars)

    def GetCurrentResult(self):
        r"""
    
        Gets the string formed by concatenating the current result list.
    
        Args:
            None
    
        Returns:
            str: The current result string.
    
        Example Output:
            "Prog"
        """
        return ''.join(self.result_chars)

    def Done(self, answer: str):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: Progamin, Reference answer: Progamin, Result: Correct, reward=1"
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
        input_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        for char in input_str:
            is_recorded = self.step(json.dumps({'name': 'CheckCharacter', 'parameters': {'character': char}}))[1]
            if is_recorded == "False":
                self.step(json.dumps({'name': 'AddCharacter', 'parameters': {'character': char}}))
        result = self.step(json.dumps({'name': 'GetCurrentResult', 'parameters': {}}))[1]
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "Programming"
    env = RemoveDuplicatesEnv.from_env_str(f"RemoveDuplicatesEnv@{{\"input_string\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "HeLloWorld"
    env = RemoveDuplicatesEnv.from_env_str(f"RemoveDuplicatesEnv@{{\"input_string\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)