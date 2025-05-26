import json
import random

# faili
input_path = "output.jsonl"
output_paths = ["lv_punct_dev.jsonl", "lv_punct_test.jsonl", "lv_punct_train.jsonl"]

with open(input_path, 'r', encoding='utf-8') as infile:
    data = [json.loads(line) for line in infile]

random.seed(42) # atkārtojamībai
random.shuffle(data) # sajauc datus

splits = [300, 300, 2400]

start = 0
for i, size in enumerate(splits): # sadala un ieraksta failos
    with open(output_paths[i], 'w', encoding='utf-8') as outfile:
        for item in data[start:start+size]:
            json_line = json.dumps(item, ensure_ascii=False)
            outfile.write(json_line + '\n')
    start += size