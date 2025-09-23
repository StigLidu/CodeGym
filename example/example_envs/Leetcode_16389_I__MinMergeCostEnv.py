import gymnasium
import ast
import json
from heapq import heappop, heappush, heapify

class MinMergeCostEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.HEAPIFY_WEIGHTS = 0
        self.POP_SMALLEST = 1
        self.MERGE_STONES = 2
        self.PUSH_TO_HEAP = 3
        self.CHECK_HEAP_SIZE = 4
        self.OBSERVE = 5
        self.DONE = 6

        # [Required] Define the action mapping
        self.func_mapping = {
            "HeapifyWeights": self.HEAPIFY_WEIGHTS,
            "PopSmallest": self.POP_SMALLEST,
            "MergeStones": self.MERGE_STONES,
            "PushToHeap": self.PUSH_TO_HEAP,
            "CheckHeapSize": self.CHECK_HEAP_SIZE,
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
        prefix = "MinMergeCostEnv@"
        if not env_str.startswith(prefix):
            return None
        return MinMergeCostEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.weights = options.get("weights", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if len(self.weights) <= 1:
            return 0
            
        heap = self.weights.copy()
        heapify(heap)
        total_cost = 0
        
        while len(heap) > 1:
            first_min = heappop(heap)
            second_min = heappop(heap)
            new_stone = first_min + second_min
            total_cost += new_stone
            heappush(heap, new_stone)
            
        return total_cost

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
            
            if action_code == self.HEAPIFY_WEIGHTS:
                if "weights" in params:
                    weights = params["weights"]
                    msg = self.HeapifyWeights(weights)
                else:
                    msg = "Error: 'weights' parameter is missing for HEAPIFY_WEIGHTS action."
            
            elif action_code == self.POP_SMALLEST:
                if "heap" in params:
                    heap = params["heap"]
                    msg = self.PopSmallest(heap)
                else:
                    msg = "Error: 'heap' parameter is missing for POP_SMALLEST action."
                    
            elif action_code == self.MERGE_STONES:
                if "a" in params and "b" in params:
                    a = params["a"]
                    b = params["b"]
                    msg = self.MergeStones(a, b)
                else:
                    msg = "Error: 'a' or 'b' parameter is missing for MERGE_STONES action."
                    
            elif action_code == self.PUSH_TO_HEAP:
                if "heap" in params and "value" in params:
                    heap = params["heap"]
                    value = params["value"]
                    msg = self.PushToHeap(heap, value)
                else:
                    msg = "Error: 'heap' or 'value' parameter is missing for PUSH_TO_HEAP action."
                    
            elif action_code == self.CHECK_HEAP_SIZE:
                if "heap" in params:
                    heap = params["heap"]
                    msg = self.CheckHeapSize(heap)
                else:
                    msg = "Error: 'heap' parameter is missing for CHECK_HEAP_SIZE action."
                    
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
    def HeapifyWeights(self, weights: list):
        r"""
    
        Convert the weight list into a min-heap.
    
        Args:
            weights (list[int]): The weight list of stones.
    
        Returns:
            str: The converted heap, serialized using json.dumps.
    
        Example Output:
            "[1, 2, 3]"
        """
        heap = weights.copy()
        heapify(heap)
        return json.dumps(heap)

    def PopSmallest(self, heap: list):
        r"""
    
        Extract the smallest element from the heap.
    
        Args:
            heap (list[int]): The min-heap.
    
        Returns:
            str: A dictionary containing the extracted element and the remaining heap, serialized using json.dumps.
    
        Example Output:
            "{\"smallest\": 1, \"remaining_heap\": [2, 3]}"
        """
        h = heap.copy()
        smallest = heappop(h)
        return json.dumps({"smallest": smallest, "remaining_heap": h})

    def MergeStones(self, a: int, b: int):
        r"""
    
        Merge two stones and return the merged weight and cost.
    
        Args:
            a (int): The weight of the first stone.
            b (int): The weight of the second stone.
    
        Returns:
            str: A dictionary containing the merged stone's weight and cost, serialized using json.dumps.
    
        Example Output:
            "{\"new_stone\": 3, \"cost\": 3}"
        """
        new_stone = a + b
        return json.dumps({"new_stone": new_stone, "cost": new_stone})

    def PushToHeap(self, heap: list, value: int):
        r"""
    
        Add a value to the heap and return the new heap.
    
        Args:
            heap (list[int]): The min-heap.
            value (int): The value to be added to the heap.
    
        Returns:
            str: The new heap after adding the value, serialized using json.dumps.
    
        Example Output:
            "[2, 3, 4]"
        """
        h = heap.copy()
        heappush(h, value)
        return json.dumps(h)

    def CheckHeapSize(self, heap: list):
        r"""
    
        Check the size of the heap.
    
        Args:
            heap (list[int]): The min-heap.
    
        Returns:
            str: The size of the heap.
    
        Example Output:
            "3"
        """
        return str(len(heap))

    def Observe(self):
        r"""
    
        Return the current list of stone weights.
    
        Args:
            None
    
        Returns:
            str: The list of stone weights, serialized using json.dumps.
    
        Example Output:
            "[3, 4, 5, 6]"
        """
        return json.dumps(self.weights)

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return result information.
    
        Args:
            answer (int): The minimum total cost submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: 36, Reference answer: 36, Result: Correct, reward=1"
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
        
        weights_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        weights = json.loads(weights_str)
        
        heap_str = self.step(json.dumps({'name': 'HeapifyWeights', 'parameters': {'weights': weights}}))[1]
        current_heap = json.loads(heap_str)
        
        total_cost = 0
        
        while True:
            size_str = self.step(json.dumps({'name': 'CheckHeapSize', 'parameters': {'heap': current_heap}}))[1]
            heap_size = int(size_str)
            if heap_size <= 1:
                break
            
            pop1_str = self.step(json.dumps({'name': 'PopSmallest', 'parameters': {'heap': current_heap}}))[1]
            pop1 = json.loads(pop1_str)
            a = pop1['smallest']
            remaining_heap1 = pop1['remaining_heap']
            
            pop2_str = self.step(json.dumps({'name': 'PopSmallest', 'parameters': {'heap': remaining_heap1}}))[1]
            pop2 = json.loads(pop2_str)
            b = pop2['smallest']
            remaining_heap2 = pop2['remaining_heap']
            
            merge_str = self.step(json.dumps({'name': 'MergeStones', 'parameters': {'a': a, 'b': b}}))[1]
            merge_result = json.loads(merge_str)
            new_stone = merge_result['new_stone']
            cost = merge_result['cost']
            total_cost += cost
            
            current_heap_str = self.step(json.dumps({'name': 'PushToHeap', 'parameters': {'heap': remaining_heap2, 'value': new_stone}}))[1]
            current_heap = json.loads(current_heap_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': total_cost}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_weights = [3, 4, 5, 6]
    env = MinMergeCostEnv.from_env_str(f"MinMergeCostEnv@{{\"weights\": {test_weights}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_weights = [1, 2, 3, 4, 5]
    env = MinMergeCostEnv.from_env_str(f"MinMergeCostEnv@{{\"weights\": {test_weights}}}")
    print(env.solve())
    print("step count:", env.step_count)