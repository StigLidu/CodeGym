# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class StringReconstructionEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.OBSERVE = 0
        self.INITIALIZE_RESULT = 1
        self.SET_CHARACTER = 2
        self.JOIN_CHARACTERS = 3
        self.DONE = 4

        # [Required] Define the action mapping
        self.func_mapping = {
            "Observe": self.OBSERVE,
            "InitializeResult": self.INITIALIZE_RESULT,
            "SetCharacter": self.SET_CHARACTER,
            "JoinCharacters": self.JOIN_CHARACTERS,
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
        prefix = "StringReconstructionEnv@"
        if not env_str.startswith(prefix):
            return None
        return StringReconstructionEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.s = options.get("s", "")
        self.indices = options.get("indices", [])
        self.result = []
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        original_word = [''] * len(self.s)
        for char, index in zip(self.s, self.indices):
            original_word[index] = char
        return ''.join(original_word)

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
            
            elif action_code == self.INITIALIZE_RESULT:
                if "length" in params:
                    length = params["length"]
                    msg = self.InitializeResult(length)
                else:
                    msg = "Error: 'length' parameter is missing for INITIALIZE_RESULT action."
            
            elif action_code == self.SET_CHARACTER:
                if "position" in params and "character" in params:
                    position = params["position"]
                    character = params["character"]
                    msg = self.SetCharacter(position, character)
                else:
                    msg = "Error: 'position' or 'character' parameter is missing for SET_CHARACTER action."
            
            elif action_code == self.JOIN_CHARACTERS:
                if "result_list" in params:
                    result_list = params["result_list"]
                    msg = self.JoinCharacters(result_list)
                else:
                    msg = "Error: 'result_list' parameter is missing for JOIN_CHARACTERS action."
            
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
    
        Obtain the shuffled string and index array in the current environment.
    
        Args:
            None
    
        Returns:
            str: A JSON string containing the shuffled string and index array.
    
        Example Output:
            "{\"s\": \"aiohn\", \"indices\": [3, 1, 4, 2, 0]}"
        """
        observation = {
            "s": self.s,
            "indices": self.indices
        }
        return json.dumps(observation)

    def InitializeResult(self, length: int):
        r"""
    
        Initialize an empty character list of a specified length for reconstructing the original word.
    
        Args:
            length (int): The length of the list to be initialized.
    
        Returns:
            str: A string representation of the initialized empty character list.
    
        Example Output:
            "['', '', '', '', '']"
        """
        self.result = [''] * length
        return str(self.result)

    def SetCharacter(self, position: int, character: str):
        r"""
    
        Place a character at the specified position in the result list.
    
        Args:
            position (int): The position index where the character is to be placed.
            character (str): The character to be placed.
    
        Returns:
            str: A string representation of the updated result list.
    
        Example Output:
            "['', 'i', '', '', '']"
        """
        if 0 <= position < len(self.result):
            self.result[position] = character
            return str(self.result)
        else:
            return f"Error: Position {position} is out of bounds for result list of length {len(self.result)}"

    def JoinCharacters(self, result_list: list):
        r"""
    
        Concatenate the character list into a single string.
    
        Args:
            result_list (list[str]): The character list to be concatenated.
    
        Returns:
            str: The concatenated string.
    
        Example Output:
            "nihao"
        """
        return ''.join(result_list)

    def Done(self, answer: str):
        r"""
    
        Verify whether the final answer is correct and return result information.
    
        Args:
            answer (str): The answer string submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: nihao, Reference answer: nihao, Result: Correct, reward=1"
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
        observe_data = json.loads(observe_result)
        s = observe_data['s']
        indices = observe_data['indices']
        length = len(s)
        init_result = self.step(json.dumps({'name': 'InitializeResult', 'parameters': {'length': length}}))[1]
        for char, idx in zip(s, indices):
            self.step(json.dumps({'name': 'SetCharacter', 'parameters': {'position': idx, 'character': char}}))
        result_list_str = init_result
        for char, idx in zip(s, indices):
            result_list_str = self.step(json.dumps({'name': 'SetCharacter', 'parameters': {'position': idx, 'character': char}}))[1]
        import ast
        result_list = ast.literal_eval(result_list_str)
        answer = self.step(json.dumps({'name': 'JoinCharacters', 'parameters': {'result_list': result_list}}))[1]
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': answer}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    s1 = "aiohn"
    indices1 = [3, 1, 4, 2, 0]
    env1 = StringReconstructionEnv.from_env_str(f"StringReconstructionEnv@{{\"s\": \"{s1}\", \"indices\": {indices1}}}")
    print(env1.solve())
    print("step count:", env1.step_count)

    # test case 2
    print("\nTest Case 2:")
    s2 = "codeleet"
    indices2 = [4,5,6,7,0,2,1,3]
    env2 = StringReconstructionEnv.from_env_str(f"StringReconstructionEnv@{{\"s\": \"{s2}\", \"indices\": {indices2}}}")
    print(env2.solve())
    print("step count:", env2.step_count)