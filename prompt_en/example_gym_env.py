import gymnasium
import random
import ast
import json

class ModeFindingEnv(gymnasium.Env):
    # Initialize the environment, please follow the format of the example
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.COUNT_OCCURRENCES = 0
        self.GET_MAX_FREQUENCY = 1
        self.GET_MODES = 2
        self.OBSERVE = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "CountOccurrences": self.COUNT_OCCURRENCES,
            "GetMaxFrequency": self.GET_MAX_FREQUENCY,
            "GetModes": self.GET_MODES,
            "Observe": self.OBSERVE,
            "Done": self.DONE
        }

        if env_str is not None:
            options = ast.literal_eval(env_str.split("@")[1])
            self.reset(options)
        else:
            self.reset()

    # [Required] Define the property and staticmethod of the environment, please follow the format of the example and remain it unchanged
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "ModeFindingEnv@"
        if not env_str.startswith(prefix):
            return None
        return ModeFindingEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.scores = options.get("scores", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Get the reference answer using information in the environment.
        """
        mode_lst = [self.scores.count(i) for i in range(11)]
        max_freq = max(mode_lst)
        return [i for i, freq in enumerate(mode_lst) if freq == max_freq]

    # [Required] Define the step method of the environment, please follow the format
    def step(self, action: str):
        r"""
        Execute an action and return result information.

        Args:
            action (str): JSON string of action name and parameters.

        Returns:
            tuple[bool, str]: Whether successfully executed, result information.
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
            
            if action_code == self.COUNT_OCCURRENCES:
                if "number" in params:
                    number = params["number"]
                    msg = self.CountOccurrences(number)
                else:
                    msg = "Error: 'number' parameter is missing for COUNT_OCCURRENCES action."
            
            elif action_code == self.GET_MAX_FREQUENCY:
                if "frequency_list" in params:
                    frequency_list = params["frequency_list"]
                    msg = self.GetMaxFrequency(frequency_list)
                else:
                    msg = "Error: 'frequency_list' parameter is missing for GET_MAX_FREQUENCY action."
            elif action_code == self.GET_MODES:
                if "frequency_list" in params and "max_freq" in params:
                    frequency_list = params["frequency_list"]
                    max_freq = params["max_freq"]
                    msg = self.GetModes(frequency_list, max_freq)
                else:
                    msg = "Error: 'frequency_list' or 'max_freq' parameter is missing for GET_MODES action."
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
    # [Required] Define the actions of the environment, each action should contain a docstring, including the args and returns
    def CountOccurrences(self, number: int):
        r"""
        Count the occurrences of number in scores.

        Args:
            number (int): The number to count.

        Returns:
            str: The count of number occurrences.

        Example Output:
            "1"
        """
        count = self.scores.count(number)
        return str(count)

    def GetMaxFrequency(self, frequency_list: list):
        r"""
        Get the maximum value in frequency_list.

        Args:
            frequency_list (list[int]): List of occurrence frequencies for each number.

        Returns:
            str: The maximum occurrence frequency.

        Example Output:
            "3"
        """
        max_freq = max(frequency_list)
        return str(max_freq)

    def GetModes(self, frequency_list: list, max_freq: int):
        r"""
        Get all numbers with occurrence frequency equal to max_freq.

        Args:
            frequency_list (list[int]): List of occurrence frequencies for each number.
            max_freq (int): Maximum occurrence frequency.

        Returns:
            str: All numbers with occurrence frequency equal to max_freq.

        Example Output:
            "[1, 2, 3]"
        """
        modes = [i for i, freq in enumerate(frequency_list) if freq == max_freq]
        return str(modes)

    def Observe(self):
        r"""
        Return current state observation information (not used in this environment, kept for interface compatibility).

        Args:
            None

        Returns:
            str: Description of current state prompt information.

        Example Output:
            "No available information, please call other actions"
        """
        return f"No available information, please call other actions"

    def Done(self, answer):
        r"""
        Verify if the final answer is correct and return result information.

        Args:
            answer (list[int]): User-submitted answer list.

        Returns:
            str: Result information, including correctness and reward information.

        Example Output:
            "Your answer: [1, 2, 3], Reference answer: [1, 2, 3], Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = sorted(answer) == sorted(ref_answer)
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # Define the solve method of the environment
    def solve(self):
        r"""
        Automatically call all actions in the environment to complete the full process and submit answer for verification.

        Returns:
            str: Final answer verification result information.
        """
        frequency_list = []
        for i in range(11):
            # call CountOccurrences
            frequency_list.append(int(self.step(json.dumps({'name': 'CountOccurrences', 'parameters': {'number': i}}))[1]))
        # call GetMaxFrequency
        max_freq = int(self.step(json.dumps({'name': 'GetMaxFrequency', 'parameters': {'frequency_list': frequency_list}}))[1])
        # call GetModes
        modes = ast.literal_eval(self.step(json.dumps({'name': 'GetModes', 'parameters': {'frequency_list': frequency_list, 'max_freq': max_freq}}))[1])
        # call Done
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': modes}}))[1]

# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_scores = [random.randint(0, 10) for _ in range(20)]
    env = ModeFindingEnv.from_env_str(f"ModeFindingEnv@{{\"scores\": {test_scores}}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("Test Case 2:")
    test_scores = [random.randint(0, 10) for _ in range(10)]
    env = ModeFindingEnv.from_env_str(f"ModeFindingEnv@{{\"scores\": {test_scores}}}")
    print(env.solve())
    print("step count:", env.step_count)
