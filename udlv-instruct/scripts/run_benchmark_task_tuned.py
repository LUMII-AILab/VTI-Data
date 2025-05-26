import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from peft import PeftModel

# modeļu mapes
# model_name = "deepseek-ai/deepseek-llm-7b-chat"
base_model_path = "./plain_models/qwen"
adapter_path = "./tuned_veryvery_models/qwen/checkpoint-2184"

# ielādē modeli, pievieno papildinošo tekstvienību ja vajag
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]', 
        'eos_token': '<|endoftext|>'
    })
else:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

with open("gold_standard.jsonl", "r", encoding="utf-8") as f:
    benchmark_data = [json.loads(line) for line in f]

# novērtešana (3x konsekvences pārbaudīšanai, ar do_sample=False īstenībā nevajag)
all_runs = []

for run_id in range(1, 4):
    print(f"iterācija {run_id}...")
    run_outputs = []

    for idx, ex in enumerate(benchmark_data):
        full_prompt = f"{ex['instruction']}\n{ex['input']}"
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)

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
        run_outputs.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "reference": ex["output"],
            "generated": generated_text.strip()
        })

    all_runs.append({
        "run": run_id,
        "outputs": run_outputs
    })

# saglabā rezultātus kā json
Path("tuned_benchmark_outputs7").mkdir(exist_ok=True)
with open("tuned_benchmark_outputs7/all_runs_qwen.json", "w", encoding="utf-8") as f:
    json.dump(all_runs, f, ensure_ascii=False, indent=2)

