import gymnasium
import ast
import json

class LexicographicalMaxEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.COMPARE_AND_POP = 1
        self.APPEND_CHAR = 2
        self.TRIM_STACK = 3
        self.JOIN_STACK = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CompareAndPop": self.COMPARE_AND_POP,
            "AppendChar": self.APPEND_CHAR,
            "TrimStack": self.TRIM_STACK,
            "JoinStack": self.JOIN_STACK,
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
        prefix = "LexicographicalMaxEnv@"
        if not env_str.startswith(prefix):
            return None
        return LexicographicalMaxEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.k = options.get("k", 0)
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        stack = []
        remaining_k = self.k
        
        for char in self.s:
            while stack and remaining_k > 0 and stack[-1] < char:
                stack.pop()
                remaining_k -= 1
            stack.append(char)
        
        if remaining_k > 0:
            stack = stack[:-remaining_k]
        
        return ''.join(stack)

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
            
            elif action_code == self.COMPARE_AND_POP:
                if "stack" in params and "char" in params and "current_k" in params:
                    stack = params["stack"]
                    char = params["char"]
                    current_k = params["current_k"]
                    msg = self.CompareAndPop(stack, char, current_k)
                else:
                    msg = "Error: 'stack', 'char' or 'current_k' parameter is missing for COMPARE_AND_POP action."
            
            elif action_code == self.APPEND_CHAR:
                if "stack" in params and "char" in params:
                    stack = params["stack"]
                    char = params["char"]
                    msg = self.AppendChar(stack, char)
                else:
                    msg = "Error: 'stack' or 'char' parameter is missing for APPEND_CHAR action."
            
            elif action_code == self.TRIM_STACK:
                if "stack" in params and "remaining_k" in params:
                    stack = params["stack"]
                    remaining_k = params["remaining_k"]
                    msg = self.TrimStack(stack, remaining_k)
                else:
                    msg = "Error: 'stack' or 'remaining_k' parameter is missing for TRIM_STACK action."
            
            elif action_code == self.JOIN_STACK:
                if "stack" in params:
                    stack = params["stack"]
                    msg = self.JoinStack(stack)
                else:
                    msg = "Error: 'stack' parameter is missing for JOIN_STACK action."
            
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
    
        Obtain the string in the current environment and the number of characters that need to be deleted.
    
        Args:
            None
    
        Returns:
            str: Observation information of the current environment, including the string and the number of characters to be deleted.
    
        Example Output:
            "{\"s\": \"abcde\", \"k\": 2}"
        """
        observation = {"s": self.s, "k": self.k}
        return json.dumps(observation)

    def CompareAndPop(self, stack: list, char: str, current_k: int):
        r"""
    
        Compare the top element of the stack with the current character. If the stack is not empty, current_k > 0, and the top element of the stack is smaller than the current character, pop the top element of the stack and decrease current_k.
    
        Args:
            stack (list): The current character stack.
            char (str): The current character to be processed.
            current_k (int): The current remaining number of characters that can be deleted.
    
        Returns:
            str: The processed stack and the remaining number of deletable characters, in JSON string format.
    
        Example Output:
            "{\"stack\": [\"a\", \"c\"], \"current_k\": 1}"
        """
        new_stack = stack.copy()
        new_k = current_k
        
        while new_stack and new_k > 0 and new_stack[-1] < char:
            new_stack.pop()
            new_k -= 1
        
        result = {"stack": new_stack, "current_k": new_k}
        return json.dumps(result)

    def AppendChar(self, stack: list, char: str):
        r"""
    
        Add the character to the end of the stack.
    
        Args:
            stack (list): The current character stack.
            char (str): The character to be added.
    
        Returns:
            str: The stack after adding the character, in JSON string format.
    
        Example Output:
            "[\"a\", \"b\", \"c\"]"
        """
        new_stack = stack.copy()
        new_stack.append(char)
        return json.dumps(new_stack)

    def TrimStack(self, stack: list, remaining_k: int):
        r"""
    
        If remaining_k > 0, delete remaining_k elements from the end of the stack.
    
        Args:
            stack (list): The current character stack.
            remaining_k (int): The remaining number of characters that can be deleted.
    
        Returns:
            str: The processed stack, in JSON string format.
    
        Example Output:
            "[\"a\", \"b\", \"c\"]"
        """
        new_stack = stack.copy()
        if remaining_k > 0:
            new_stack = new_stack[:-remaining_k]
        return json.dumps(new_stack)

    def JoinStack(self, stack: list):
        r"""
    
        Concatenate the characters in the stack into a string.
    
        Args:
            stack (list): The current character stack.
    
        Returns:
            str: The concatenated string.
    
        Example Output:
            "abc"
        """
        return ''.join(stack)

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: cde, Reference answer: cde, Result: Correct, reward=1"
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
        observe_result = self.step(json.dumps({"name": "Observe", "parameters": {}}))[1]
        observe_data = json.loads(observe_result)
        s = observe_data["s"]
        k = observe_data["k"]
        
        stack = []
        current_k = k
        
        for char in s:
            cap_params = {
                "stack": stack,
                "char": char,
                "current_k": current_k
            }
            cap_result = self.step(json.dumps({"name": "CompareAndPop", "parameters": cap_params}))[1]
            cap_data = json.loads(cap_result)
            stack = cap_data["stack"]
            current_k = cap_data["current_k"]
            
            append_params = {
                "stack": stack,
                "char": char
            }
            append_result = self.step(json.dumps({"name": "AppendChar", "parameters": append_params}))[1]
            stack = json.loads(append_result)
        
        trim_params = {
            "stack": stack,
            "remaining_k": current_k
        }
        trim_result = self.step(json.dumps({"name": "TrimStack", "parameters": trim_params}))[1]
        stack = json.loads(trim_result)
        
        join_result = self.step(json.dumps({"name": "JoinStack", "parameters": {"stack": stack}}))[1]
        
        return self.step(json.dumps({"name": "Done", "parameters": {"answer": join_result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    env1 = LexicographicalMaxEnv.from_env_str('LexicographicalMaxEnv@{"s": "abcde", "k": 2}')
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    env2 = LexicographicalMaxEnv.from_env_str('LexicographicalMaxEnv@{"s": "bacdb", "k": 2}')
    print(env2.solve())
    print("step count:", env2.step_count)