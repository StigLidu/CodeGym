# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

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