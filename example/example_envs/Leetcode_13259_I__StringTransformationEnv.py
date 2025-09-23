import gymnasium
import ast
import json

class StringTransformationEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CALCULATE_CHAR_DIFF = 0
        self.SUM_DIFFERENCES = 1
        self.CHECK_LENGTH = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CalculateCharDiff": self.CALCULATE_CHAR_DIFF,
            "SumDifferences": self.SUM_DIFFERENCES,
            "CheckLength": self.CHECK_LENGTH,
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
        prefix = "StringTransformationEnv@"
        if not env_str.startswith(prefix):
            return None
        return StringTransformationEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s1 = options.get("s1", "")
        self.s2 = options.get("s2", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if len(self.s1) != len(self.s2):
            return -1
            
        operations_needed = 0
        for i in range(len(self.s1)):
            diff = (ord(self.s2[i]) - ord(self.s1[i])) % 26
            operations_needed += diff
        
        return operations_needed

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
            
            if action_code == self.CALCULATE_CHAR_DIFF:
                if "char1" in params and "char2" in params:
                    char1 = params["char1"]
                    char2 = params["char2"]
                    msg = self.CalculateCharDiff(char1, char2)
                else:
                    msg = "Error: 'char1' or 'char2' parameter is missing for CALCULATE_CHAR_DIFF action."
            
            elif action_code == self.SUM_DIFFERENCES:
                if "differences" in params:
                    differences = params["differences"]
                    msg = self.SumDifferences(differences)
                else:
                    msg = "Error: 'differences' parameter is missing for SUM_DIFFERENCES action."
                    
            elif action_code == self.CHECK_LENGTH:
                if "str1" in params and "str2" in params:
                    str1 = params["str1"]
                    str2 = params["str2"]
                    msg = self.CheckLength(str1, str2)
                else:
                    msg = "Error: 'str1' or 'str2' parameter is missing for CHECK_LENGTH action."
                    
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
    def CalculateCharDiff(self, char1: str, char2: str):
        r"""
    
        Calculate the minimum number of increment operations required to convert character char1 to char2.
    
        Args:
            char1 (str): Original character, a single lowercase letter.
            char2 (str): Target character, a single lowercase letter.
    
        Returns:
            str: The minimum number of operations needed for the conversion.
    
        Example Output:
            "2"
        """
        diff = (ord(char2) - ord(char1)) % 26
        return str(diff)

    def SumDifferences(self, differences: list):
        r"""
    
        Calculate the sum of all elements in the difference list.
    
        Args:
            differences (list[int]): A list containing multiple difference values.
    
        Returns:
            str: The sum of the difference list.
    
        Example Output:
            "6"
        """
        total = sum(differences)
        return str(total)

    def CheckLength(self, str1: str, str2: str):
        r"""
    
        Check if the lengths of two strings are the same.
    
        Args:
            str1 (str): The first string.
            str2 (str): The second string.
    
        Returns:
            str: Returns "True" if the lengths are the same, otherwise returns "False".
    
        Example Output:
            "True"
        """
        return str(len(str1) == len(str2))

    def Observe(self):
        r"""
    
        Obtain the two strings s1 and s2 in the current environment.
    
        Args:
            None
    
        Returns:
            str: Information containing s1 and s2.
    
        Example Output:
            "s1: abc, s2: cde"
        """
        return f"s1: {self.s1}, s2: {self.s2}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the minimum number of operations or -1.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 6, Reference answer: 6, Result: Correct, reward=1"
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
        observe_result = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        parts = observe_result.split(', ')
        s1 = parts[0].split(': ')[1]
        s2 = parts[1].split(': ')[1]
        
        length_check = self.step(json.dumps({'name': 'CheckLength', 'parameters': {'str1': s1, 'str2': s2}}))[1]
        if length_check == "False":
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': -1}}))[1]
        
        differences = []
        for c1, c2 in zip(s1, s2):
            diff = int(self.step(json.dumps({'name': 'CalculateCharDiff', 'parameters': {'char1': c1, 'char2': c2}}))[1])
            differences.append(diff)
        
        total = int(self.step(json.dumps({'name': 'SumDifferences', 'parameters': {'differences': differences}}))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = StringTransformationEnv.from_env_str('StringTransformationEnv@{"s1": "abc", "s2": "cde"}')
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = StringTransformationEnv.from_env_str('StringTransformationEnv@{"s1": "xyz", "s2": "abc"}')
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3 (different lengths)
    print("\nTest Case 3:")
    env = StringTransformationEnv.from_env_str('StringTransformationEnv@{"s1": "ab", "s2": "abc"}')
    print(env.solve())
    print("step count:", env.step_count)