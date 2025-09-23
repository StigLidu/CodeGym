import os
import json
import sys

folder_path = sys.argv[1]
target_file = sys.argv[2]

for file in os.listdir(folder_path):
    if file.endswith('.json'):
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)
            with open(target_file, 'a') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')