import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# deepseek-ai/deepseek-llm-7b-chat
# google/gemma-3-12b-it
# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Ministral-8B-Instruct-2410
# Qwen/Qwen2.5-14B-Instruct

# modeļa mape
save_dir = "./plain_models/qwen"

tokenizer = AutoTokenizer.from_pretrained(save_dir)
tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})  # pievieno eos tekstvienību lai mēģinātu apturēt izvadi
model = AutoModelForCausalLM.from_pretrained(save_dir, device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model.eval()

with open("gold_standard.jsonl", "r", encoding="utf-8") as f:
    benchmark_data = [json.loads(line) for line in f]

all_runs = []

# novērtešana (3x konsekvences pārbaudīšanai, ar do_sample=False īstenībā nevajag)
for run_id in range(1, 4):
    print(f"iterācija {run_id}...")

    run_outputs = []

    for idx, ex in enumerate(benchmark_data):
        prompt = f"{ex['instruction']}\n{ex['input']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # parametri ģenerēšanai
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # causes temp warning, irrelevant
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # mēģina tikt vaļā no atkārtotā vaicājuma izvades...
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        run_outputs.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "reference": ex["output"],
            "generated": generated_text.strip()
        })

    all_runs.append({"run": run_id, "outputs": run_outputs})

# saglabā kā json lasāmībai
Path("benchmark_outputs").mkdir(exist_ok=True)
with open("benchmark_outputs/all_runs_qwen.json", "w", encoding="utf-8") as f:
    json.dump(all_runs, f, ensure_ascii=False, indent=2)
