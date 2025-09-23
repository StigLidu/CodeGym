import gymnasium
import ast
import json
from collections import Counter

class TargetFormationEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.CHECK_PREFIX_MATCH = 1
        self.DECREASE_WORD_COUNT = 2
        self.INCREASE_WORD_COUNT = 3
        self.CHECK_EMPTY_STRING = 4
        self.GET_WORD_COUNTER = 5
        self.GET_SUBSTRING = 6
        self.DONE = 7

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "CheckPrefixMatch": self.CHECK_PREFIX_MATCH,
            "DecreaseWordCount": self.DECREASE_WORD_COUNT,
            "IncreaseWordCount": self.INCREASE_WORD_COUNT,
            "CheckEmptyString": self.CHECK_EMPTY_STRING,
            "GetWordCounter": self.GET_WORD_COUNTER,
            "GetSubstring": self.GET_SUBSTRING,
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
        prefix = "TargetFormationEnv@"
        if not env_str.startswith(prefix):
            return None
        return TargetFormationEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.words = options.get("words", [])
        self.target = options.get("target", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def can_form_recursively(tgt, counter):
            if tgt == "":
                return True
            for word in list(counter.keys()):
                if tgt.startswith(word) and counter[word] > 0:
                    counter[word] -= 1
                    if can_form_recursively(tgt[len(word):], counter):
                        return True
                    counter[word] += 1
            return False
        
        word_counter = Counter(self.words)
        return can_form_recursively(self.target, word_counter)

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
                
            elif action_code == self.CHECK_PREFIX_MATCH:
                if "word" in params and "target_substring" in params:
                    word = params["word"]
                    target_substring = params["target_substring"]
                    msg = self.CheckPrefixMatch(word, target_substring)
                else:
                    msg = "Error: 'word' or 'target_substring' parameter is missing for CHECK_PREFIX_MATCH action."
                    
            elif action_code == self.DECREASE_WORD_COUNT:
                if "word" in params and "counter" in params:
                    word = params["word"]
                    counter = params["counter"]
                    msg = self.DecreaseWordCount(word, counter)
                else:
                    msg = "Error: 'word' or 'counter' parameter is missing for DECREASE_WORD_COUNT action."
                    
            elif action_code == self.INCREASE_WORD_COUNT:
                if "word" in params and "counter" in params:
                    word = params["word"]
                    counter = params["counter"]
                    msg = self.IncreaseWordCount(word, counter)
                else:
                    msg = "Error: 'word' or 'counter' parameter is missing for INCREASE_WORD_COUNT action."
                    
            elif action_code == self.CHECK_EMPTY_STRING:
                if "s" in params:
                    s = params["s"]
                    msg = self.CheckEmptyString(s)
                else:
                    msg = "Error: 's' parameter is missing for CHECK_EMPTY_STRING action."
                    
            elif action_code == self.GET_WORD_COUNTER:
                msg = self.GetWordCounter()
                
            elif action_code == self.GET_SUBSTRING:
                if "s" in params and "start" in params and "end" in params:
                    s = params["s"]
                    start = params["start"]
                    end = params["end"]
                    msg = self.GetSubstring(s, start, end)
                else:
                    msg = "Error: 's', 'start' or 'end' parameter is missing for GET_SUBSTRING action."
                    
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
    
        Obtain the list of words and the target string in the current environment.
    
        Args:
            None
    
        Returns:
            str: Information containing the list of words and the target string.
    
        Example Output:
            "{\"words\": [\"ab\", \"cd\"], \"target\": \"abcd\"}"
        """
        return json.dumps({"words": self.words, "target": self.target})

    def CheckPrefixMatch(self, word: str, target_substring: str):
        r"""
    
        Check if the word is a prefix of the target substring.
    
        Args:
            word (str): The word to be checked.
            target_substring (str): The target substring.
    
        Returns:
            str: "True" if the word is a prefix of the target substring, otherwise "False".
    
        Example Output:
            "True"
        """
        return str(target_substring.startswith(word))

    def DecreaseWordCount(self, word: str, counter: dict):
        r"""
    
        Decrease the count of the specified word in the counter.
    
        Args:
            word (str): The word whose count is to be decreased.
            counter (dict): The word count dictionary.
    
        Returns:
            str: The updated count dictionary.
    
        Example Output:
            "{\"ab\": 1, \"cd\": 0}"
        """
        if word in counter:
            counter[word] -= 1
        return json.dumps(counter)

    def IncreaseWordCount(self, word: str, counter: dict):
        r"""
    
        Increase the count of the specified word in the counter.
    
        Args:
            word (str): The word whose count is to be increased.
            counter (dict): The word count dictionary.
    
        Returns:
            str: The updated count dictionary.
    
        Example Output:
            "{\"ab\": 2, \"cd\": 1}"
        """
        if word in counter:
            counter[word] += 1
        return json.dumps(counter)

    def CheckEmptyString(self, s: str):
        r"""
    
        Check if the string is empty.
    
        Args:
            s (str): The string to be checked.
    
        Returns:
            str: "True" if the string is empty, otherwise "False".
    
        Example Output:
            "True"
        """
        return str(s == "")

    def GetWordCounter(self):
        r"""
    
        Obtain the count of each word in the word list.
    
        Args:
            None
    
        Returns:
            str: The word count dictionary.
    
        Example Output:
            "{\"ab\": 1, \"cd\": 1}"
        """
        counter = Counter(self.words)
        return json.dumps(dict(counter))

    def GetSubstring(self, s: str, start: int, end: int):
        r"""
    
        Obtain the substring of a string.
    
        Args:
            s (str): The original string.
            start (int): The starting index of the substring.
            end (int): The ending index of the substring.
    
        Returns:
            str: The substring from start to end.
    
        Example Output:
            "cd"
        """
        return s[start:end]

    def Done(self, answer: bool):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (bool): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: True, Reference answer: True, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
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
        
        obs = self.step(json.dumps({"name": "Observe", "parameters": {}}))[1]
        obs_dict = json.loads(obs)
        words = obs_dict["words"]
        target = obs_dict["target"]
        
        counter_str = self.step(json.dumps({"name": "GetWordCounter", "parameters": {}}))[1]
        counter = json.loads(counter_str)
        
        def backtrack(remaining_target, current_counter):
            is_empty = self.step(json.dumps({
                "name": "CheckEmptyString",
                "parameters": {"s": remaining_target}
            }))[1]
            if is_empty == "True":
                return True
            
            for word in words:
                if current_counter.get(word, 0) <= 0:
                    continue
                
                is_prefix = self.step(json.dumps({
                    "name": "CheckPrefixMatch",
                    "parameters": {"word": word, "target_substring": remaining_target}
                }))[1]
                if is_prefix != "True":
                    continue
                
                updated_counter_str = self.step(json.dumps({
                    "name": "DecreaseWordCount",
                    "parameters": {"word": word, "counter": current_counter}
                }))[1]
                updated_counter = json.loads(updated_counter_str)
                
                word_len = len(word)
                new_remaining = self.step(json.dumps({
                    "name": "GetSubstring",
                    "parameters": {"s": remaining_target, "start": word_len, "end": len(remaining_target)}
                }))[1]
                
                if backtrack(new_remaining, updated_counter):
                    return True
                
                current_counter = json.loads(self.step(json.dumps({
                    "name": "IncreaseWordCount",
                    "parameters": {"word": word, "counter": updated_counter}
                }))[1])
            
            return False
        
        result = backtrack(target, counter)
        
        return self.step(json.dumps({"name": "Done", "parameters": {"answer": result}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_words1 = ["ab", "cd", "ab"]
    test_target1 = "abcdab"
    env1 = TargetFormationEnv.from_env_str(
        f"TargetFormationEnv@{{\"words\": {test_words1}, \"target\": \"{test_target1}\"}}"
    )
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_words2 = ["a", "bc"]
    test_target2 = "abcd"
    env2 = TargetFormationEnv.from_env_str(
        f"TargetFormationEnv@{{\"words\": {test_words2}, \"target\": \"{test_target2}\"}}"
    )
    print(env2.solve())
    print("step count:", env2.step_count)