from transformers import AutoTokenizer, AutoModelForCausalLM
import os

print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
print("HF_HOME:", os.getenv("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))
print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))
print("TMPDIR:", os.getenv("TMPDIR"))

# deepseek-ai/deepseek-llm-7b-chat
# google/gemma-3-12b-it
# meta-llama/Llama-3.1-8B-Instruct
# mistralai/Ministral-8B-Instruct-2410
# Qwen/Qwen2.5-7B-Instruct

model_name = "Qwen/Qwen2.5-7B-Instruct"
save_dir = "./plain_models/qwen"

# lejupielādē modeli
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# un saglabā
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
