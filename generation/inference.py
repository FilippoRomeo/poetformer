from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# Load tokenizer from base model
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = os.path.abspath("../models/lora")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter using absolute path
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


def generate_poem(prompt, max_new_tokens=120, temperature=0.9, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
