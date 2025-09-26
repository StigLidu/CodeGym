# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
from openai import OpenAI
import requests

def format_messages(messages):
    if type(messages) == str:
        messages = [{"role": "system", "content": messages}]
    elif type(messages) == list and len(messages) > 0 and type(messages[0]) == str:
        for i in range(len(messages)):
            if i % 2 == 0:
                messages[i] = {"role": "user", "content": messages[i]}
            else:
                messages[i] = {"role": "assistant", "content": messages[i]}
    return messages

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai_llm(prompt, price_calculate=False, num=1):
    prompt = format_messages(prompt)
    completion = client.chat.completions.create(
        n=num,
        model="gpt-4.1",
        messages=prompt,
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
    )
    # price calculation
    # for gpt-4.1, the price is 2 USD per 1M input tokens and 8 USD per 1M output tokens
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    price = (input_tokens / 1000000) * 2 + (output_tokens / 1000000) * 8
    output_list = []
    for i in range(num):
        output_list.append(completion.choices[i].message.content)
    if price_calculate:
        return output_list, price
    else:
        return output_list

if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello?"}]
    response = call_openai_llm(messages)
    print(response)