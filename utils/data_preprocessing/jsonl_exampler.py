# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

# show the first line, convert to json format
import json
import sys
file_path = sys.argv[1]
if len(sys.argv) > 2:
    idx = int(sys.argv[2])
else:
    idx = 0
with open(file_path, "r") as f:
    for i in range(idx + 1):
        line = f.readline()
    line = json.loads(line)
    print(line.keys())
    if "response" in line:
        print(line["response"])
    else:
        print(json.dumps(line, ensure_ascii=False, indent=2))

# save the top-10 liens
if len(sys.argv) > 3:
    select_prefix_num = int(sys.argv[3])
else:
    select_prefix_num = 10
with open(file_path, "r") as f:
    line_10 = []
    for i in range(select_prefix_num):
        line = f.readline()
        line = json.loads(line)
        line_10.append(line)
    with open(file_path.replace(".jsonl", f"_top_{select_prefix_num}.jsonl"), "w") as f:
        for line in line_10:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

# print the total number of entries
with open(file_path, "r") as f:
    total_num = 0
    for line in f:
        total_num += 1
    print(f"Total number of entries: {total_num}")