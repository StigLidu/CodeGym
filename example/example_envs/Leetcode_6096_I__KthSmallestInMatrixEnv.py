import gymnasium
import heapq
import ast
import json
from typing import List

class KthSmallestInMatrixEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.INITIALIZE_HEAP = 0
        self.POP_FROM_HEAP = 1
        self.PUSH_TO_HEAP = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "InitializeHeap": self.INITIALIZE_HEAP,
            "PopFromHeap": self.POP_FROM_HEAP,
            "PushToHeap": self.PUSH_TO_HEAP,
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
        prefix = "KthSmallestInMatrixEnv@"
        if not env_str.startswith(prefix):
            return None
        return KthSmallestInMatrixEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.matrix = options.get("matrix", [[]])
        self.k = options.get("k", 1)
        self.min_heap = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        n = len(self.matrix)
        min_heap = []
        
        for r in range(min(n, self.k)):
            heapq.heappush(min_heap, (self.matrix[r][0], r, 0))
        
        val = None
        for _ in range(self.k):
            val, r, c = heapq.heappop(min_heap)
            if c + 1 < n:
                heapq.heappush(min_heap, (self.matrix[r][c + 1], r, c + 1))
        
        return val

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
            
            if action_code == self.INITIALIZE_HEAP:
                if "limit" in params:
                    limit = params["limit"]
                    msg = self.InitializeHeap(limit)
                else:
                    msg = "Error: 'limit' parameter is missing for INITIALIZE_HEAP action."
            
            elif action_code == self.POP_FROM_HEAP:
                msg = self.PopFromHeap()
                
            elif action_code == self.PUSH_TO_HEAP:
                if "row" in params and "col" in params:
                    row = params["row"]
                    col = params["col"]
                    msg = self.PushToHeap(row, col)
                else:
                    msg = "Error: 'row' or 'col' parameter is missing for PUSH_TO_HEAP action."
                    
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
    def InitializeHeap(self, limit: int):
        r"""
    
        Initialize a min-heap, add the first element of each row of the matrix to the heap, adding at most 'limit' elements.
        
        Args:
            limit (int): The maximum number of elements to be added to the heap.
            
        Returns:
            str: A prompt message indicating the completion of heap initialization, including the size of the heap.
            
        Example Output:
            "Heap initialized with 3 elements"
        """
        n = len(self.matrix)
        self.min_heap = []
        for r in range(min(n, limit)):
            heapq.heappush(self.min_heap, (self.matrix[r][0], r, 0))
        return f"Heap initialized with {len(self.min_heap)} elements"

    def PopFromHeap(self):
        r"""
    
        Pop the smallest element from the min-heap.
        
        Args:
            None
            
        Returns:
            str: Information of the popped element, formatted as "value,row,col".
            
        Example Output:
            "5,0,1"
        """
        if not self.min_heap:
            return "Error: Heap is empty"
            
        val, r, c = heapq.heappop(self.min_heap)
        return f"{val},{r},{c}"

    def PushToHeap(self, row: int, col: int):
        r"""
    
        Add the element at the specified position in the matrix to the min-heap.
        
        Args:
            row (int): Row index of the element.
            col (int): Column index of the element.
            
        Returns:
            str: A prompt message indicating the element has been added to the heap, including the element's value.
            
        Example Output:
            "Pushed element 11 to heap"
        """
        n = len(self.matrix)
        if row < 0 or row >= n or col < 0 or col >= n:
            return "Error: Invalid row or column index"
            
        value = self.matrix[row][col]
        heapq.heappush(self.min_heap, (value, row, col))
        return f"Pushed element {value} to heap"

    def Observe(self):
        r"""
    
        Return the observation information of the current environment, including the matrix size and the value of k.
        
        Args:
            None
            
        Returns:
            str: Observation information of the current environment.
            
        Example Output:
            "n x n matrix where n=3, k=8"
        """
        n = len(self.matrix)
        return f"n x n matrix where n={n}, k={self.k}"

    def Done(self, answer: int):
        r"""
    
        Verify whether the final answer is correct and return the result information.
        
        Args:
            answer (int): The answer submitted by the user, i.e., the k-th smallest element.
            
        Returns:
            str: Result information, including whether it is correct and reward information.
            
        Example Output:
            "Your answer: 13, Reference answer: 13, Result: Correct, reward=1"
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
        n = int(observe_info.split('n=')[1].split(',')[0])
        k = int(observe_info.split('k=')[1])
        
        self.step(json.dumps({'name': 'InitializeHeap', 'parameters': {'limit': n}}))
        
        result = 0
        for _ in range(k):
            pop_info = self.step(json.dumps({'name': 'PopFromHeap', 'parameters': {}}))[1]
            value, row, col = pop_info.split(',')
            result = int(value)
            row = int(row)
            col = int(col)
            
            if col + 1 < n:
                self.step(json.dumps({'name': 'PushToHeap', 'parameters': {'row': row, 'col': col + 1}}))
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - example from problem
    print("Test Case 1:")
    matrix1 = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    k1 = 8
    env1 = KthSmallestInMatrixEnv.from_env_str(f"KthSmallestInMatrixEnv@{{\"matrix\": {matrix1}, \"k\": {k1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2 - another example
    print("\nTest Case 2:")
    matrix2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    k2 = 5
    env2 = KthSmallestInMatrixEnv.from_env_str(f"KthSmallestInMatrixEnv@{{\"matrix\": {matrix2}, \"k\": {k2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)