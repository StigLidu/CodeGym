# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import gymnasium
import ast
import json

class AtlantisCodeDecoderEnv(gymnasium.Env):
    def __init__(self, env_str: str=None):
        super().__init__()

        # [Required] Define the action names
        self.DECODE_CHARACTER = 0
        self.COMBINE_CHARACTERS = 1
        self.OBSERVE = 2
        self.DONE = 3

        # [Required] Define the action mapping
        self.func_mapping = {
            "DecodeCharacter": self.DECODE_CHARACTER,
            "CombineCharacters": self.COMBINE_CHARACTERS,
            "Observe": self.OBSERVE,
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
        prefix = "AtlantisCodeDecoderEnv@"
        if not env_str.startswith(prefix):
            return None
        return AtlantisCodeDecoderEnv(env_str=env_str)

    # [Required] Define the reset method of the environment
    def reset(self, options={}):
        self.message = options.get("message", "")
        self._reward = 0
        self._done = False
        self.step_count = 0
        return "Environment has been reset."

    # [Required] Get the reference answer of the environment
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        vowel_to_number = {
            'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5',
            'A': '1', 'E': '2', 'I': '3', 'O': '4', 'U': '5'
        }
        
        decoded_message = []
        for char in self.message:
            if char in vowel_to_number:
                decoded_message.append(vowel_to_number[char])
            else:
                decoded_message.append(char)
        
        return ''.join(decoded_message)

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
            
            if action_code == self.DECODE_CHARACTER:
                if "char" in params:
                    char = params["char"]
                    msg = self.DecodeCharacter(char)
                else:
                    msg = "Error: 'char' parameter is missing for DECODE_CHARACTER action."
            
            elif action_code == self.COMBINE_CHARACTERS:
                if "characters" in params:
                    characters = params["characters"]
                    msg = self.CombineCharacters(characters)
                else:
                    msg = "Error: 'characters' parameter is missing for COMBINE_CHARACTERS action."
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
    def DecodeCharacter(self, char: str):
        r"""
    
        Decode a single character. If it is a vowel, return the corresponding number; otherwise, return the original character.
    
        Args:
            char (str): The single character to be decoded.
    
        Returns:
            str: The decoded character or number.
    
        Example Output:
            "2"
        """
        vowel_to_number = {
            'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5',
            'A': '1', 'E': '2', 'I': '3', 'O': '4', 'U': '5'
        }
        return vowel_to_number.get(char, char)

    def CombineCharacters(self, characters: list):
        r"""
    
        Combine the list of characters into a string.
    
        Args:
            characters (list[str]): The list of characters to be combined.
    
        Returns:
            str: The combined string.
    
        Example Output:
            "h2LL4"
        """
        return ''.join(characters)

    def Observe(self):
        r"""
    
        Return the current message that needs to be decoded.
    
        Args:
            None
    
        Returns:
            str: The current message to be decoded.
    
        Example Output:
            "heLLo"
        """
        return self.message

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (str): The decoding result submitted by the user.
    
        Returns:
            str: Result information, including whether it is correct and reward information.
    
        Example Output:
            "Your answer: h2LL4, Reference answer: h2LL4, Result: Correct, reward=1"
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
        encoded_message = self.step(json.dumps({'name': 'Observe', 'parameters': {}}))[1]
        decoded_chars = []
        for char in encoded_message:
            decoded_char = self.step(json.dumps({'name': 'DecodeCharacter', 'parameters': {'char': char}}))[1]
            decoded_chars.append(decoded_char)
        decoded_string = self.step(json.dumps({'name': 'CombineCharacters', 'parameters': {'characters': decoded_chars}}))[1]
        return self.step(json.dumps({'name': 'Done', 'parameters': {'answer': decoded_string}}))[1]
# Test the environment
if __name__ == "__main__":
    # test case 1
    print("Test Case 1:")
    test_message = "heLLo"
    env = AtlantisCodeDecoderEnv.from_env_str(f"AtlantisCodeDecoderEnv@{{\"message\": \"{test_message}\"}}")
    print(env.solve())
    print("step count:", env.step_count)

    # test case 2
    print("\nTest Case 2:")
    test_message = "cOdevIllE"
    env = AtlantisCodeDecoderEnv.from_env_str(f"AtlantisCodeDecoderEnv@{{\"message\": \"{test_message}\"}}")
    print(env.solve())
    print("step count:", env.step_count)