import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# modeļu mapes
base_model_path = "./plain_models/qwen"
adapter_path = "./tuned_veryvery_models/qwen/checkpoint-2184"
model_id = "qwen"
eval_file = "lv_punct_test.jsonl"

# ielādē modeļus + pielāgoto
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def normalize_line(s): # noņem eos tekstvienību
    return s.strip().replace("<|endoftext|>", "").strip()

with open(eval_file, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f]

preds = []
refs = []
outputs = []
mistakes = []

total_correct_lines = 0
total_lines = 0
exact_match_count = 0
example_count = 0

# novērtēšana
for ex in eval_data:
    example_count += 1
    prompt = f"{ex['instruction']}\n{ex['input']}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # parametri ģenerēšanai
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # causes temp warning, irrelevant
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = output_ids[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    ref_text = ex["output"].strip()

    preds.append(generated_text)
    refs.append(ref_text)

    # salīdzina rindas
    ref_lines = ref_text.splitlines()
    pred_lines = generated_text.splitlines()
    line_matches = [normalize_line(r) == normalize_line(p) for r, p in zip(ref_lines, pred_lines)]
    correct = sum(line_matches)
    total = len(ref_lines)

    total_correct_lines += correct
    total_lines += total

    is_exact_match = normalize_line(ref_text) == normalize_line(gen_text)

    if is_exact_match:
        exact_match_count += 1

    if example_count % 25 == 0:
        print(f"currently on example {example_count}...")

    example = {
        "instruction": ex["instruction"],
        "input": ex["input"],
        "reference": ref_text,
        "generated": generated_text,
    }

    # pieraksta kļūdainās izvades, ja gribas iziet tam cauri
    if not is_exact_match:
        wrong_lines = []
        for i, (r, p) in enumerate(zip(ref_lines, pred_lines)):
            if normalize_line(r) != normalize_line(p):
                wrong_lines.append({
                "line_num": i + 1,
                "expected": r.strip(),
                "got": p.strip()
            })
        example["mistake_lines"] = wrong_lines
        mistakes.append(example)

    outputs.append(example)

# metrikas
exact_match = exact_match_count / len(eval_data)
line_accuracy = total_correct_lines / total_lines if total_lines else 0
total_mistakes = total_lines - total_correct_lines

Path("eval_outputs7").mkdir(exist_ok=True)

with open(f"eval_outputs7/eval_{model_id}.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_id": model_id,
        "exact_match": exact_match,
        "exact_match_percent": f"{exact_match * 100:.2f}%",
        "global_line_accuracy": line_accuracy,
        "line_accuracy_percent": f"{line_accuracy * 100:.2f}%",
        "total_examples": len(eval_data),
        "total_example_mistakes": len(mistakes),
        "total_line_mistakes": f"{total_mistakes} / {total_lines}",
        "examples": outputs,
        "mistakes": mistakes
    }, f, ensure_ascii=False, indent=2)


# python3 eval_script.py | tee eval_deepseek3.log