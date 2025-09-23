import gymnasium
import ast
import json

class ParenthesesValidationEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.PROCESS_CHARACTER = 0
        self.CALCULATE_ADD_NEEDED = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "ProcessCharacter": self.PROCESS_CHARACTER,
            "CalculateAddNeeded": self.CALCULATE_ADD_NEEDED,
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
        prefix = "ParenthesesValidationEnv@"
        if not env_str.startswith(prefix):
            return None
        return ParenthesesValidationEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.current_index = 0
        self.balance = 0
        self.add_needed = 0
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        balance = 0
        add_needed = 0
        
        for char in self.s:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                
            if balance < 0:
                add_needed += 1
                balance = 0
                
        return add_needed + balance

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
                if "char" in params and "current_balance" in params and "current_add_needed" in params:
                    char = params["char"]
                    current_balance = params["current_balance"]
                    current_add_needed = params["current_add_needed"]
                    msg = self.ProcessCharacter(char, current_balance, current_add_needed)
                else:
                    msg = "Error: 'char', 'current_balance' or 'current_add_needed' parameter is missing for PROCESS_CHARACTER action."
            
            elif action_code == self.CALCULATE_ADD_NEEDED:
                if "final_balance" in params and "final_add_needed" in params:
                    final_balance = params["final_balance"]
                    final_add_needed = params["final_add_needed"]
                    msg = self.CalculateAddNeeded(final_balance, final_add_needed)
                else:
                    msg = "Error: 'final_balance' or 'final_add_needed' parameter is missing for CALCULATE_ADD_NEEDED action."
                    
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
    def ProcessCharacter(self, char: str, current_balance: int, current_add_needed: int):
        r"""
    
        Process a parenthesis character and update the parenthesis balance status and the number of needed parentheses to add.
    
        Args:
            char (str): The parenthesis character to be processed, which can only be '(' or ')'.
            current_balance (int): The current parenthesis balance value.
            current_add_needed (int): The currently calculated number of parentheses that need to be added.
    
        Returns:
            str: A JSON string containing the updated balance value and the addition requirement.
    
        Example Output:
            "{\"balance\": 1, \"add_needed\": 0}"
        """
        if char == '(':
            current_balance += 1
        elif char == ')':
            current_balance -= 1
            
        if current_balance < 0:
            current_add_needed += 1
            current_balance = 0
            
        return json.dumps({"balance": current_balance, "add_needed": current_add_needed})

    def CalculateAddNeeded(self, final_balance: int, final_add_needed: int):
        r"""
    
        Calculate the total number of parentheses that need to be added finally.
    
        Args:
            final_balance (int): The final balance value after processing all characters.
            final_add_needed (int): The accumulated addition requirement during processing.
    
        Returns:
            str: The total number of parentheses that need to be added finally.
    
        Example Output:
            "2"
        """
        total_add = final_add_needed + final_balance
        return str(total_add)

    def Observe(self):
        r"""
    
        Obtain the parenthesis string in the current environment.
    
        Args:
            None
    
        Returns:
            str: The parenthesis string in the current environment.
    
        Example Output:
            "())("
        """
        return self.s

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The answer submitted by the user, i.e., the minimum number of parentheses that need to be added.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: 2, Reference answer: 2, Result: Correct, reward=1"
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
        current_balance = 0
        current_add_needed = 0
        for char in s:
            result = self.step(json.dumps({
                'name': 'ProcessCharacter',
                'parameters': {
                    'char': char,
                    'current_balance': current_balance,
                    'current_add_needed': current_add_needed
                }
            }))[1]
            result_dict = json.loads(result)
            current_balance = result_dict['balance']
            current_add_needed = result_dict['add_needed']
        final_answer = int(self.step(json.dumps({
            'name': 'CalculateAddNeeded',
            'parameters': {
                'final_balance': current_balance,
                'final_add_needed': current_add_needed
            }
        }))[1])
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': final_answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_str = "())"
    env = ParenthesesValidationEnv.from_env_str(f"ParenthesesValidationEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_str = "((("
    env = ParenthesesValidationEnv.from_env_str(f"ParenthesesValidationEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)
    
    # test case 3
    print("\nTest Case 3:")
    test_str = "()()"
    env = ParenthesesValidationEnv.from_env_str(f"ParenthesesValidationEnv@{{\"s\": \"{test_str}\"}}")
    print(env.solve())
    print("step count:", env.step_count)