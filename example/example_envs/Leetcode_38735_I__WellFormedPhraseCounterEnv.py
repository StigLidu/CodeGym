import gymnasium
import ast
import json

class WellFormedPhraseCounterEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.CHECK_LEADING_TRAILING = 0
        self.SPLIT_INTO_WORDS = 1
        self.CHECK_WORD_FORMAT = 2
        self.COUNT_WELL_FORMED = 3
        self.OBSERVE = 4
        self.DONE = 5

        # [Required] Define the action mapping
        self.func_mapping = {
            "CheckLeadingTrailingSpaces": self.CHECK_LEADING_TRAILING,
            "SplitIntoWords": self.SPLIT_INTO_WORDS,
            "CheckWordFormat": self.CHECK_WORD_FORMAT,
            "CountWellFormed": self.COUNT_WELL_FORMED,
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
        prefix = "WellFormedPhraseCounterEnv@"
        if not env_str.startswith(prefix):
            return None
        return WellFormedPhraseCounterEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.phrases = options.get("phrases", [])
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        def is_well_formed(phrase):
            if phrase.startswith(' ') or phrase.endswith(' '):
                return False
            words = phrase.split(' ')
            return all(word.islower() and word.isalpha() for word in words)
        
        return sum(1 for phrase in self.phrases if is_well_formed(phrase))

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
            
            if action_code == self.CHECK_LEADING_TRAILING:
                if "phrase" in params:
                    phrase = params["phrase"]
                    msg = self.CheckLeadingTrailingSpaces(phrase)
                else:
                    msg = "Error: 'phrase' parameter is missing for CHECK_LEADING_TRAILING action."
            
            elif action_code == self.SPLIT_INTO_WORDS:
                if "phrase" in params:
                    phrase = params["phrase"]
                    msg = self.SplitIntoWords(phrase)
                else:
                    msg = "Error: 'phrase' parameter is missing for SPLIT_INTO_WORDS action."
            
            elif action_code == self.CHECK_WORD_FORMAT:
                if "word" in params:
                    word = params["word"]
                    msg = self.CheckWordFormat(word)
                else:
                    msg = "Error: 'word' parameter is missing for CHECK_WORD_FORMAT action."
            
            elif action_code == self.COUNT_WELL_FORMED:
                if "flags" in params:
                    flags = params["flags"]
                    msg = self.CountWellFormed(flags)
                else:
                    msg = "Error: 'flags' parameter is missing for COUNT_WELL_FORMED action."
            
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
    def CheckLeadingTrailingSpaces(self, phrase: str):
        r"""
    
        Check if the phrase has leading or trailing spaces.
    
        Args:
            phrase (str): The phrase to be checked.
    
        Returns:
            str: Returns "True" if there are leading or trailing spaces, otherwise returns "False".
    
        Example Output:
            "False"
        """
        has_bad_spaces = phrase.startswith(' ') or phrase.endswith(' ')
        return str(has_bad_spaces)

    def SplitIntoWords(self, phrase: str):
        r"""
    
        Split the phrase into a list of words by spaces.
    
        Args:
            phrase (str): The phrase to be split.
    
        Returns:
            str: The string representation of the split list of words.
    
        Example Output:
            "['hello', 'world']"
        """
        words = phrase.split(' ')
        return str(words)

    def CheckWordFormat(self, word: str):
        r"""
    
        Check if the word contains only lowercase letters.
    
        Args:
            word (str): The word to be checked.
    
        Returns:
            str: Returns "True" if the word contains only lowercase letters, otherwise returns "False".
    
        Example Output:
            "True"
        """
        is_valid = word.islower() and word.isalpha()
        return str(is_valid)

    def CountWellFormed(self, flags: list):
        r"""
    
        Count the number of well-formed phrases.
    
        Args:
            flags (list[bool]): A list of flags indicating whether each phrase is well-formed.
    
        Returns:
            str: The number of well-formed phrases.
    
        Example Output:
            "2"
        """
        count = sum(1 for flag in flags if flag)
        return str(count)

    def Observe(self):
        r"""
    
        Get the list of phrases in the current environment.
    
        Args:
            None
    
        Returns:
            str: The string representation of the list of phrases.
    
        Example Output:
            "['hello world', '  leading space', 'trailing space  ', 'well formed', 'not  well']"
        """
        return str(self.phrases)

    def Done(self, answer):
        r"""
    
        Verify if the final answer is correct and return result information.
    
        Args:
            answer (int): The answer submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
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
        import ast
        phrases_str = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        phrases = ast.literal_eval(phrases_str)
        flags = []
        
        for phrase in phrases:
            has_bad_spaces = self.step(json.dumps({
                'name': 'CheckLeadingTrailingSpaces',
                'parameters': {'phrase': phrase}
            }))[1]
            if has_bad_spaces == "True":
                flags.append(False)
                continue
            
            words_str = self.step(json.dumps({
                'name': 'SplitIntoWords',
                'parameters': {'phrase': phrase}
            }))[1]
            words = ast.literal_eval(words_str)
            
            valid = True
            for word in words:
                is_valid = self.step(json.dumps({
                    'name': 'CheckWordFormat',
                    'parameters': {'word': word}
                }))[1]
                if is_valid == "False":
                    valid = False
                    break
            
            flags.append(valid)
        
        count = self.step(json.dumps({
            'name': 'CountWellFormed',
            'parameters': {'flags': flags}
        }))[1]
        
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': int(count)}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_phrases1 = ["hello world", "  leading space", "trailing space  ", "well formed", "not  well"]
    env1 = WellFormedPhraseCounterEnv.from_env_str(f"WellFormedPhraseCounterEnv@{{\"phrases\": {test_phrases1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_phrases2 = ["valid phrase", "   extra   spaces   ", "Mixed Case", "singleword", ""]
    env2 = WellFormedPhraseCounterEnv.from_env_str(f"WellFormedPhraseCounterEnv@{{\"phrases\": {test_phrases2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)