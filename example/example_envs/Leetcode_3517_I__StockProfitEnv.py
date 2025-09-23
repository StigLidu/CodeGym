import gymnasium
import ast
import json
from typing import List

class StockProfitEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.FIND_MIN_PRICE = 1
        self.CALCULATE_PROFIT = 2
        self.FIND_MAX_PROFIT = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "FindMinPrice": self.FIND_MIN_PRICE,
            "CalculateProfit": self.CALCULATE_PROFIT,
            "FindMaxProfit": self.FIND_MAX_PROFIT,
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
        prefix = "StockProfitEnv@"
        if not env_str.startswith(prefix):
            return None
        return StockProfitEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.prices = options.get("prices", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self) -> int:
        r"""
        Use the information in the environment to get the reference answer. 
        """
        min_price = float('inf')
        max_profit = 0
        
        for price in self.prices:
            if price < min_price:
                min_price = price
            profit = price - min_price
            if profit > max_profit:
                max_profit = profit
        
        return max_profit

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
            
            elif action_code == self.FIND_MIN_PRICE:
                if "start_index" in params:
                    start_index = params["start_index"]
                    msg = self.FindMinPrice(start_index)
                else:
                    msg = "Error: 'start_index' parameter is missing for FIND_MIN_PRICE action."
            
            elif action_code == self.CALCULATE_PROFIT:
                if "buy_price" in params and "sell_price" in params:
                    buy_price = params["buy_price"]
                    sell_price = params["sell_price"]
                    msg = self.CalculateProfit(buy_price, sell_price)
                else:
                    msg = "Error: 'buy_price' or 'sell_price' parameter is missing for CALCULATE_PROFIT action."
            
            elif action_code == self.FIND_MAX_PROFIT:
                if "profit_list" in params:
                    profit_list = params["profit_list"]
                    msg = self.FindMaxProfit(profit_list)
                else:
                    msg = "Error: 'profit_list' parameter is missing for FIND_MAX_PROFIT action."
            
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
    
        Obtain the current list of stock prices.
    
        Args:
            None
    
        Returns:
            str: A string representation of the list of stock prices.
    
        Example Output:
            "[7, 1, 5, 3, 6, 4]"
        """
        return str(self.prices)

    def FindMinPrice(self, start_index: int):
        r"""
    
        Starting from the specified start index, find the minimum price in the price list.
    
        Args:
            start_index (int): The start index.
    
        Returns:
            str: The minimum price.
    
        Example Output:
            "1"
        """
        if start_index < 0 or start_index >= len(self.prices):
            return "Error: Invalid start_index"
        min_price = min(self.prices[start_index:])
        return str(min_price)

    def CalculateProfit(self, buy_price: int, sell_price: int):
        r"""
    
        Calculate the profit from buying at buy_price and selling at sell_price.
    
        Args:
            buy_price (int): The buying price.
            sell_price (int): The selling price.
    
        Returns:
            str: The profit value, returns 0 if negative.
    
        Example Output:
            "5"
        """
        profit = sell_price - buy_price
        return str(max(0, profit))

    def FindMaxProfit(self, profit_list: List[int]):
        r"""
    
        Find the maximum profit value from the profit list.
    
        Args:
            profit_list (List[int]): The profit list.
    
        Returns:
            str: The maximum profit value.
    
        Example Output:
            "5"
        """
        if not profit_list:
            return "0"
        return str(max(profit_list))

    def Done(self, answer: int):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (int): The user-submitted maximum profit answer.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        prices_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        prices = ast.literal_eval(prices_str)
        if len(prices) < 2:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        profit_list = []
        for i in range(len(prices) - 1):
            buy_price = prices[i]
            for j in range(i + 1, len(prices)):
                sell_price = prices[j]
                profit = int(self.step(json.dumps({
                    'name': 'CalculateProfit',
                    'parameters': {'buy_price': buy_price, 'sell_price': sell_price}
                }))[1])
                profit_list.append(profit)
        
        if not profit_list:
            max_profit = 0
        else:
            max_profit = int(self.step(json.dumps({
                'name': 'FindMaxProfit',
                'parameters': {'profit_list': profit_list}
            }))[1])
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_profit}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_prices1 = [7, 1, 5, 3, 6, 4]
    env1 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_prices2 = [7, 6, 4, 3, 1]
    env2 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)