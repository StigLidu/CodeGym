import gymnasium
import ast
import json

class MatrixBeautyEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_IF_POSSIBLE = 1
        self.COUNT_POSSIBLE_VALUES = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckIfPossible": self.CHECK_IF_POSSIBLE,
            "CountPossibleValues": self.COUNT_POSSIBLE_VALUES,
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
        prefix = "MatrixBeautyEnv@"
        if not env_str.startswith(prefix):
            return None
        return MatrixBeautyEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.n = options.get("n", 1)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        return self.n

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
            
            elif action_code == self.CHECK_IF_POSSIBLE:
                if "k" in params:
                    k = params["k"]
                    msg = self.CheckIfPossible(k)
                else:
                    msg = "Error: 'k' parameter is missing for CHECK_IF_POSSIBLE action."
                    
            elif action_code == self.COUNT_POSSIBLE_VALUES:
                msg = self.CountPossibleValues()
                
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
    
        Obtain the value of n in the current environment.
    
        Args:
            None
    
        Returns:
            str: The current value of n.
    
        Example Output:
            "5"
        """
        return str(self.n)

    def CheckIfPossible(self, k: int):
        r"""
    
        Check if the integer k can be represented as the aesthetic value of a certain matrix.
    
        Args:
            k (int): The integer to be checked.
    
        Returns:
            str: "True" indicates it can be represented, "False" indicates it cannot be represented.
    
        Example Output:
            "True"
        """
        return "True" if k >= 1 else "False"

    def CountPossibleValues(self):
        r"""
    
        Calculate how many integers from 1 to n can be represented as the aesthetic value of a matrix.
    
        Args:
            None
    
        Returns:
            str: The number of integers that can be represented as the aesthetic value of a matrix.
    
        Example Output:
            "5"
        """
        return str(self.n)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
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
        n_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        n = int(n_str)
        
        count = 0
        for k in range(1, n + 1):
            check_result = self.step(json.dumps({'name': 'CheckIfPossible', 'parameters': {'k': k}}))[1]
            if check_result == "True":
                count += 1
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': count}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env = MatrixBeautyEnv.from_env_str("MatrixBeautyEnv@{\"n\": 3}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = MatrixBeautyEnv.from_env_str("MatrixBeautyEnv@{\"n\": 7}")
    print(env.solve())
    print("step count:", env.step_count)