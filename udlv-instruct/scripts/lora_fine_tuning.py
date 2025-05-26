import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
print("HF_HOME:", os.getenv("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))
print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))
print("TMPDIR:", os.getenv("TMPDIR"))

# deepseek-ai/deepseek-llm-7b-chat
# google/gemma-3-12b-it
# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Ministral-8B-Instruct-2410
# Qwen/Qwen2.5-14B-Instruct

# modeļu mapes un treniņdati
model_path = "./plain_models/mistral/"
output_dir = "./tuned_veryvery_models/mistral/"
train_file = "lv_punct_train.jsonl"
eval_file = "lv_punct_dev.jsonl"

# ielādē modeli
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

# pievieno papildinošo tekstvienību, ja tādas nav
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    print("Adding a distinct [PAD] token")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",  # Gemma3 vajag
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# LoRA konfigurācija
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={"train": train_file, "validation": eval_file})

tokenizer.padding_side = "right"

# tokenizācija
def tokenize(example):
    text = f"{example['instruction']}\n{example['input']}\n{example['output']}<|endoftext|>"
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # noņem nost visas papildinošās tekstvienības
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    tokenized["labels"] = labels
    return tokenized

tokenized = dataset.map(tokenize)

sample = tokenized["train"][0]
print("Input IDs: ", sample["input_ids"])
print("Labels: ", sample["labels"])
print("Valid Label Tokens: ", [x for x in sample["labels"] if x != -100])

print(tokenizer.convert_tokens_to_ids("<|endoftext|>")) # pārbauda eos tekstvienību
print(tokenizer.decode([tokenizer.eos_token_id])) # tas pats šeit jo bija problēmas

# hiperparametri
args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=25,
    num_train_epochs=7,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True, # visiem modeļiem šādi labāk
    fp16=False,
    save_total_limit=5,
    report_to="tensorboard",
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator
)

# pielāgošana
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(":) hope it's better now")

# python3 lora_fine_tuning.py | tee ./logs/fine_tune_mistral7.log
