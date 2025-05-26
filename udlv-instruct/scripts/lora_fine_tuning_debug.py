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
model_name = "deepseek-ai/deepseek-llm-7b-chat" 
model_path = "./plain_models/deepseek/"
output_dir = "./tuned_models/deepseek/"
train_file = "train_debug.jsonl"
eval_file = "dev_debug.jsonl"

# ielādē modeli
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.resize_token_embeddings(len(tokenizer))

# LoRA konfigurācija
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={"train": train_file, "validation": eval_file})

# atšķir papildinošo no eos tekstvienības
tokenizer.pad_token = tokenizer.eos_token
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
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = dataset.map(tokenize)

print(tokenizer.convert_tokens_to_ids("<|endoftext|>")) # pārbauda eos tekstvienību
print(tokenizer.decode([tokenizer.eos_token_id])) # tas pats šeit jo bija problēmas

# hiperparametri
args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="no",  # atkļūdošanā nevajag
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=1,
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    weight_decay=0.0,
    fp16=False,
    report_to="tensorboard",
    load_best_model_at_end=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
