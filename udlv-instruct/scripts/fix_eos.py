import json

#palīgfunkcija, lai pievienotu eos tekstvienību treniņdatu output rindās – lai iemācītu modeļiem

def add_eos_to_jsonl(input_path, output_path, eos_token="<|endoftext|>"): 
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            sample = json.loads(line)
            if not sample["output"].strip().endswith(eos_token):
                sample["output"] = sample["output"].rstrip() + eos_token
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

add_eos_to_jsonl("train.jsonl", "lv_punct_train.jsonl")
add_eos_to_jsonl("dev.jsonl", "lv_punct_dev.jsonl")
add_eos_to_jsonl("test.jsonl", "lv_punct_test.jsonl")
