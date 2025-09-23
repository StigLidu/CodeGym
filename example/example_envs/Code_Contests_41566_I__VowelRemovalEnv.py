import gymnasium
import ast
import json

class VowelRemovalEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.IS_VOWEL = 1
        self.BUILD_RESULT = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "IsVowel": self.IS_VOWEL,
            "BuildResult": self.BUILD_RESULT,
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
        prefix = "VowelRemovalEnv@"
        if not env_str.startswith(prefix):
            return None
        return VowelRemovalEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.input_string = options.get("input_string", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        vowels = 'aeiouAEIOU'
        return ''.join([char for char in self.input_string if char not in vowels])

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
            
            elif action_code == self.IS_VOWEL:
                if "char" in params:
                    char = params["char"]
                    msg = self.IsVowel(char)
                else:
                    msg = "Error: 'char' parameter is missing for IS_VOWEL action."
                    
            elif action_code == self.BUILD_RESULT:
                if "result_so_far" in params and "char" in params:
                    result_so_far = params["result_so_far"]
                    char = params["char"]
                    msg = self.BuildResult(result_so_far, char)
                else:
                    msg = "Error: 'result_so_far' or 'char' parameter is missing for BUILD_RESULT action."
                    
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
    
        Obtain the input string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The input string in the current environment.
    
        Example Output:
            "banana"
        """
        return self.input_string

    def IsVowel(self, char: str):
        r"""
    
        Determine whether the given character is a vowel (a, e, i, o, u, case-insensitive).
    
        Args:
            char (str): The single character to be judged.
    
        Returns:
            str: "True" indicates it is a vowel, "False" indicates it is not a vowel.
    
        Example Output:
            "True"
        """
        vowels = 'aeiouAEIOU'
        return str(char in vowels)

    def BuildResult(self, result_so_far: str, char: str):
        r"""
    
        Add a character to the result string to build a new result.
    
        Args:
            result_so_far (str): The currently built result string.
            char (str): The character to be added to the result.
    
        Returns:
            str: The new result string after adding the character.
    
        Example Output:
            "bn"
        """
        return result_so_far + char

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: bnn, Reference answer: bnn, Result: Correct, reward=1"
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
        result = ""
        for char in input_str:
            is_vowel = self.step(json.dumps({'name': 'IsVowel', 'parameters': {'char': char}}))[1]
            if is_vowel == "False":
                result = self.step(json.dumps({'name': 'BuildResult', 'parameters': {'result_so_far': result, 'char': char}}))[1]
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_string = "banana"
    env = VowelRemovalEnv.from_env_str(f"VowelRemovalEnv@{{\"input_string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_string = "Hello World!"
    env = VowelRemovalEnv.from_env_str(f"VowelRemovalEnv@{{\"input_string\": \"{test_string}\"}}")
    print(env.solve())
    print("step count:", env.step_count)