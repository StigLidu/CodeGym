# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import json
import ast
import random

class StockProfitEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.FIND_MIN_PRICE = 1
        self.CALCULATE_PROFIT = 2
        self.UPDATE_MAX_PROFIT = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "FindMinPrice": self.FIND_MIN_PRICE,
            "CalculateProfit": self.CALCULATE_PROFIT,
            "UpdateMaxProfit": self.UPDATE_MAX_PROFIT,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property of the environment
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
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        if not self.prices or len(self.prices) < 2:
            return 0
        
        min_price = self.prices[0]
        max_profit = 0
        
        for price in self.prices[1:]:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
        
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
                if "start_idx" in params and "end_idx" in params:
                    start_idx = params["start_idx"]
                    end_idx = params["end_idx"]
                    msg = self.FindMinPrice(start_idx, end_idx)
                else:
                    msg = "Error: 'start_idx' or 'end_idx' parameter is missing for FIND_MIN_PRICE action."
            
            elif action_code == self.CALCULATE_PROFIT:
                if "buy_price" in params and "sell_price" in params:
                    buy_price = params["buy_price"]
                    sell_price = params["sell_price"]
                    msg = self.CalculateProfit(buy_price, sell_price)
                else:
                    msg = "Error: 'buy_price' or 'sell_price' parameter is missing for CALCULATE_PROFIT action."
            
            elif action_code == self.UPDATE_MAX_PROFIT:
                if "current_max" in params and "new_profit" in params:
                    current_max = params["current_max"]
                    new_profit = params["new_profit"]
                    msg = self.UpdateMaxProfit(current_max, new_profit)
                else:
                    msg = "Error: 'current_max' or 'new_profit' parameter is missing for UPDATE_MAX_PROFIT action."
            
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
    
        Returns the current list of stock prices.
    
        Args:
            None
    
        Returns:
            str: A string representation of the list of stock prices.
    
        Example Output:
            "[7, 1, 5, 3, 6, 4]"
        """
        return str(self.prices)

    def FindMinPrice(self, start_idx: int, end_idx: int):
        r"""
    
        Finds the minimum price in the range from start_idx to end_idx (inclusive) in the price list.
    
        Args:
            start_idx (int): Start index
            end_idx (int): End index
    
        Returns:
            str: A string representation of the minimum price.
    
        Example Output:
            "1"
        """
        if start_idx < 0 or end_idx >= len(self.prices) or start_idx > end_idx:
            return "0"
        return str(min(self.prices[start_idx:end_idx+1]))

    def CalculateProfit(self, buy_price: int, sell_price: int):
        r"""
    
        Calculates the profit from buying at buy_price and selling at sell_price.
    
        Args:
            buy_price (int): Purchase price
            sell_price (int): Selling price
    
        Returns:
            str: A string representation of the profit; returns "0" if the profit is negative.
    
        Example Output:
            "5"
        """
        profit = sell_price - buy_price
        return str(max(0, profit))

    def UpdateMaxProfit(self, current_max: int, new_profit: int):
        r"""
    
        Compares the current maximum profit with the new profit and returns the larger value.
    
        Args:
            current_max (int): Current maximum profit
            new_profit (int): Newly calculated profit
    
        Returns:
            str: The updated maximum profit.
    
        Example Output:
            "5"
        """
        return str(max(current_max, new_profit))

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns the result information.
    
        Args:
            answer (int): The user-submitted answer for the maximum profit.
    
        Returns:
            str: Result information, including whether it is correct and reward details.
    
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
        n = len(prices)
        if n < 2:
            return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': 0}}))[1]
        
        max_profit = 0
        for buy_idx in range(n - 1):
            sell_start = buy_idx + 1
            sell_end = n - 1
            
            buy_price = prices[buy_idx]
            
            for sell_idx in range(buy_idx + 1, n):
                sell_price = prices[sell_idx]
                profit_str = self.step(json.dumps({
                    'name': 'CalculateProfit',
                    'parameters': {'buy_price': buy_price, 'sell_price': sell_price}
                }))[1]
                profit = int(profit_str)
                max_profit_str = self.step(json.dumps({
                    'name': 'UpdateMaxProfit',
                    'parameters': {'current_max': max_profit, 'new_profit': profit}
                }))[1]
                max_profit = int(max_profit_str)
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': max_profit}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1 - from sample input 1
    print("Test Case 1:")
    test_prices1 = [7, 1, 5, 3, 6, 4]
    env1 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)
    
    # test case 2 - from sample input 2
    print("\nTest Case 2:")
    test_prices2 = [7, 6, 4, 3, 1]
    env2 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)
    
    # test case 3 - from sample input 3
    print("\nTest Case 3:")
    test_prices3 = [1, 2, 3, 4, 5, 6, 7]
    env3 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices3}}}")
    print(env3.solve())
    print("step count:", env3.step_count)
    
    # test case 4 - random prices
    print("\nTest Case 4:")
    test_prices4 = [random.randint(0, 100) for _ in range(random.randint(2, 10))]
    env4 = StockProfitEnv.from_env_str(f"StockProfitEnv@{{\"prices\": {test_prices4}}}")
    print(f"Prices: {test_prices4}")
    print(env4.solve())
    print("step count:", env4.step_count)