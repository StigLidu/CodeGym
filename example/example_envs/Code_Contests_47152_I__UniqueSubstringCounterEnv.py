import gymnasium
import ast
import json

class UniqueSubstringCounterEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_SUBSTRING = 0
        self.GET_STRING_LENGTH = 1
        self.INCREMENT_COUNTER = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckSubstring": self.CHECK_SUBSTRING,
            "GetStringLength": self.GET_STRING_LENGTH,
            "IncrementCounter": self.INCREMENT_COUNTER,
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
        prefix = "UniqueSubstringCounterEnv@"
        if not env_str.startswith(prefix):
            return None
        return UniqueSubstringCounterEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.k = options.get("k", 0)
        self.genetic_code = options.get("genetic_code", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        strlen = len(self.genetic_code)
        if self.k > strlen:
            return 0
        
        count = 0
        for i in range(strlen - self.k + 1):
            substring = self.genetic_code[i:i+self.k]
            if len(set(substring)) == self.k:
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
            
            if action_code == self.CHECK_SUBSTRING:
                if "index" in params and "k" in params:
                    index = params["index"]
                    k_val = params["k"]
                    msg = self.CheckSubstring(index, k_val)
                else:
                    msg = "Error: 'index' or 'k' parameter is missing for CHECK_SUBSTRING action."
            
            elif action_code == self.GET_STRING_LENGTH:
                msg = self.GetStringLength()
                
            elif action_code == self.INCREMENT_COUNTER:
                if "counter" in params:
                    counter = params["counter"]
                    msg = self.IncrementCounter(counter)
                else:
                    msg = "Error: 'counter' parameter is missing for INCREMENT_COUNTER action."
                    
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
    def CheckSubstring(self, index: int, k: int):
        r"""
    
        Check if the substring of length k starting from the specified index contains all unique characters.
    
        Args:
            index (int): The starting index of the substring.
            k (int): The length of the substring.
    
        Returns:
            str: "True" if the substring contains all unique characters, otherwise "False".
    
        Example Output:
            "True"
        """
        if index + k > len(self.genetic_code):
            return "False"
        substring = self.genetic_code[index:index+k]
        return "True" if len(set(substring)) == k else "False"

    def GetStringLength(self):
        r"""
    
        Get the length of the genetic code string.
    
        Args:
            None
    
        Returns:
            str: The length of the genetic code string.
    
        Example Output:
            "7"
        """
        return str(len(self.genetic_code))

    def IncrementCounter(self, counter: int):
        r"""
    
        Increment the counter by 1.
    
        Args:
            counter (int): The current value of the counter.
    
        Returns:
            str: The value of the counter after incrementing.
    
        Example Output:
            "3"
        """
        return str(counter + 1)

    def Observe(self):
        r"""
    
        Return the observation information of the current environment, including the k value and genetic code.
    
        Args:
            None
    
        Returns:
            str: Observation information containing the k value and genetic code.
    
        Example Output:
            "k=3, genetic_code=abacdec"
        """
        return f"k={self.k}, genetic_code={self.genetic_code}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 4, Reference answer: 4, Result: Correct, reward=1"
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
        observe_info = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        k = int(observe_info.split(',')[0].split('=')[1].strip())
        str_length = int(self.step(json.dumps({'name': 'GetStringLength', 'parameters': {}}))[1])
        max_index = str_length - k
        if max_index < 0:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        counter = 0
        for index in range(max_index + 1):
            is_valid = self.step(json.dumps({'name': 'CheckSubstring', 'parameters': {'index': index, 'k': k}}))[1]
            if is_valid == "True":
                counter = int(self.step(json.dumps({'name': 'IncrementCounter', 'parameters': {'counter': counter}}))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': counter}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 (sample input)
    print("Test Case 1:")
    env = UniqueSubstringCounterEnv.from_env_str('UniqueSubstringCounterEnv@{"k": 3, "genetic_code": "abacdec"}')
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    env = UniqueSubstringCounterEnv.from_env_str('UniqueSubstringCounterEnv@{"k": 2, "genetic_code": "abcabc"}')
    print(env.solve())
    print("step count:", env.step_count)