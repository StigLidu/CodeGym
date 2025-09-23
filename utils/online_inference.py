import os
import sys
import json
from tqdm import tqdm
from utils.utils import load_jsonl_or_parquet
from utils.parquet_utils import list_of_dicts_to_parquet
from utils.api import call_openai_llm

def online_inference(input_file, input_key, output_key, num = 1, repeat = 1):
    input_data = load_jsonl_or_parquet(input_file)
    for data in tqdm(input_data):
        prompt = data[input_key]
        total_output = []
        for i in range(repeat):
            output = call_openai_llm(prompt, num = num)
            total_output.extend(output)
        data[output_key] = total_output

    output_file = input_file.replace(".parquet", f"_{output_key}.parquet").replace(".jsonl", f"_{output_key}.jsonl")
    # add a suffix to the output file
    if input_file.endswith(".parquet"):
        list_of_dicts_to_parquet(input_data, output_file)
    elif input_file.endswith(".jsonl"):
        with open(output_file, "w") as f:
            for item in input_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported data format: {input_file}")