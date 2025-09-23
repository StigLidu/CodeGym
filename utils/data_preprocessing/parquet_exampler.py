import pandas as pd
import sys
import json

from utils.utils import load_jsonl_or_parquet

# 1. 读取 parquet 文件（替换成你自己的文件路径）
df = load_jsonl_or_parquet(sys.argv[1])

# 4. parquet 的数据总数
data_count = 0
save_data = []
for record in df:
    data_count += 1
    if data_count == 10:
        print(record)
    if data_count <= 10:
        save_data.append(record)
print(f"parquet 数据总数: {data_count}")

# 3. 保存前 10 行为 JSONL 格式
output_path = sys.argv[1].replace(".parquet", "_top-10.jsonl")
with open(output_path, 'w', encoding='utf-8') as f:
    for record in save_data:
        json_line = json.dumps(record, ensure_ascii=False)
        f.write(json_line + '\n')