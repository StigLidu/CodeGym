import pandas as pd
import json
import sys

# 1) Read the Parquet file
file_name = sys.argv[1]
df = pd.read_parquet(file_name)

# 2) Write out as JSON‐lines, preserving Unicode
df.to_json(
    file_name.replace(".parquet", ".jsonl"),
    orient='records',   # one JSON object per row
    lines=True,         # newline-delimited (“JSONL”)
    force_ascii=False   # keep non-ASCII chars
)

# read the first line
with open(file_name.replace(".parquet", ".jsonl"), 'r') as f:
    first_line = json.loads(f.readline())
    # format print
    print(json.dumps(first_line, ensure_ascii=False, indent=4))