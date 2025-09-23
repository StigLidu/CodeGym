import gymnasium
import ast
import json

class LargestRectangleAreaEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.PUSH_TO_STACK = 1
        self.POP_FROM_STACK = 2
        self.PROCESS_REMAINING_STACK = 3
        self.UPDATE_MAX_AREA = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "PushToStack": self.PUSH_TO_STACK,
            "PopFromStack": self.POP_FROM_STACK,
            "ProcessRemainingStack": self.PROCESS_REMAINING_STACK,
            "UpdateMaxArea": self.UPDATE_MAX_AREA,
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
        prefix = "LargestRectangleAreaEnv@"
        if not env_str.startswith(prefix):
            return None
        return LargestRectangleAreaEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.heights = options.get("heights", [])
        self.stack = []
        self.current_index = 0
        self.max_area = 0
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
        max_area = 0
        index = 0
        
        while index < len(self.heights):
            if not stack or self.heights[stack[-1]] <= self.heights[index]:
                stack.append(index)
                index += 1
            else:
                top_of_stack = stack.pop()
                area = (self.heights[top_of_stack] *
                       ((index - stack[-1] - 1) if stack else index))
                max_area = max(max_area, area)
        
        while stack:
            top_of_stack = stack.pop()
            area = (self.heights[top_of_stack] *
                  ((index - stack[-1] - 1) if stack else index))
            max_area = max(max_area, area)
        
        return max_area

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
            
            elif action_code == self.PUSH_TO_STACK:
                if "index" in params:
                    index = params["index"]
                    msg = self.PushToStack(index)
                else:
                    msg = "Error: 'index' parameter is missing for PUSH_TO_STACK action."
            
            elif action_code == self.POP_FROM_STACK:
                if "index" in params:
                    index = params["index"]
                    msg = self.PopFromStack(index)
                else:
                    msg = "Error: 'index' parameter is missing for POP_FROM_STACK action."
            
            elif action_code == self.PROCESS_REMAINING_STACK:
                if "index" in params:
                    index = params["index"]
                    msg = self.ProcessRemainingStack(index)
                else:
                    msg = "Error: 'index' parameter is missing for PROCESS_REMAINING_STACK action."
            
            elif action_code == self.UPDATE_MAX_AREA:
                if "current_max" in params and "new_area" in params:
                    current_max = params["current_max"]
                    new_area = params["new_area"]
                    msg = self.UpdateMaxArea(current_max, new_area)
                else:
                    msg = "Error: 'current_max' or 'new_area' parameter is missing for UPDATE_MAX_AREA action."
            
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
    
        Obtain the height list of the current histogram and the current index.
    
        Args:
            None
    
        Returns:
            str: Information containing the histogram height list and the current index.
    
        Example Output:
            "heights: [2,1,5,6,2,3], current_index: 0"
        """
        return f"heights: {self.heights}, current_index: {self.current_index}, stack: {self.stack}, max_area: {self.max_area}"

    def PushToStack(self, index: int):
        r"""
    
        Push the specified index onto the stack and increment the current index.
    
        Args:
            index (int): The index value to be pushed onto the stack.
    
        Returns:
            str: The operation result, including the current stack state and the new index.
    
        Example Output:
            "Stack after push: [0], new index: 1"
        """
        self.stack.append(index)
        self.current_index = index + 1
        return f"Stack after push: {self.stack}, new index: {self.current_index}"

    def PopFromStack(self, index: int):
        r"""
    
        Pop the top element from the stack and calculate the area of the rectangle with that element as the height.
    
        Args:
            index (int): The current traversed index.
    
        Returns:
            str: The operation result, including the popped index and the calculated area.
    
        Example Output:
            "Popped index: 3, calculated area: 6"
        """
        if not self.stack:
            return "Error: Stack is empty"
            
        top_of_stack = self.stack.pop()
        if self.stack:
            width = index - self.stack[-1] - 1
        else:
            width = index
        area = self.heights[top_of_stack] * width
        return f"Popped index: {top_of_stack}, calculated area: {area}"

    def ProcessRemainingStack(self, index: int):
        r"""
    
        Process the remaining elements in the stack and calculate the area of the rectangle with each popped element as the height.
    
        Args:
            index (int): The index value after traversal ends.
    
        Returns:
            str: The operation result, including the popped index and the calculated area.
    
        Example Output:
            "Popped index: 2, calculated area: 10"
        """
        if not self.stack:
            return "Error: Stack is empty"
            
        top_of_stack = self.stack.pop()
        if self.stack:
            width = index - self.stack[-1] - 1
        else:
            width = index
        area = self.heights[top_of_stack] * width
        return f"Popped index: {top_of_stack}, calculated area: {area}"

    def UpdateMaxArea(self, current_max: int, new_area: int):
        r"""
    
        Update the maximum area if the newly calculated area is greater than the current maximum.
    
        Args:
            current_max (int): The current maximum area.
            new_area (int): The newly calculated area.
    
        Returns:
            str: The operation result, including the updated maximum area.
    
        Example Output:
            "Max area updated from 6 to 10"
        """
        self.max_area = max(current_max, new_area)
        return f"Max area updated from {current_max} to {self.max_area}"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The user-submitted answer for the maximum rectangular area.
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: 10, Reference answer: 10, Result: Correct, reward=1"
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
        heights_str = observe_result.split('heights: ')[1].split(', current_index:')[0]
        heights = eval(heights_str)
        n = len(heights)
        current_index = 0
        max_area = 0
        stack = []
        
        while current_index < n:
            if not stack:
                push_result = self.step(json.dumps({
                    'name': 'PushToStack',
                    'parameters': {'index': current_index}
                }))[1]
                current_index += 1
                stack = eval(push_result.split('Stack after push: ')[1].split(', new index:')[0])
            else:
                top_index = stack[-1]
                if heights[current_index] >= heights[top_index]:
                    push_result = self.step(json.dumps({
                        'name': 'PushToStack',
                        'parameters': {'index': current_index}
                    }))[1]
                    current_index += 1
                    stack = eval(push_result.split('Stack after push: ')[1].split(', new index:')[0])
                else:
                    pop_result = self.step(json.dumps({
                        'name': 'PopFromStack',
                        'parameters': {'index': current_index}
                    }))[1]
                    new_area = int(pop_result.split('calculated area: ')[1])
                    update_result = self.step(json.dumps({
                        'name': 'UpdateMaxArea',
                        'parameters': {'current_max': max_area, 'new_area': new_area}
                    }))[1]
                    max_area = int(update_result.split('to ')[1])
                    stack.pop()
        
        while stack:
            pop_result = self.step(json.dumps({
                'name': 'ProcessRemainingStack',
                'parameters': {'index': current_index}
            }))[1]
            new_area = int(pop_result.split('calculated area: ')[1])
            update_result = self.step(json.dumps({
                'name': 'UpdateMaxArea',
                'parameters': {'current_max': max_area, 'new_area': new_area}
            }))[1]
            max_area = int(update_result.split('to ')[1])
            stack.pop()
        
        return self.step(json.dumps({
            'name': 'Done',
            'parameters': {'answer': max_area}
        }))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_heights = [2,1,5,6,2,3]
    env = LargestRectangleAreaEnv.from_env_str(f"LargestRectangleAreaEnv@{{\"heights\": {test_heights}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_heights = [1,2,3,4,5]
    env = LargestRectangleAreaEnv.from_env_str(f"LargestRectangleAreaEnv@{{\"heights\": {test_heights}}}")
    print(env.solve())
    print("step count:", env.step_count)