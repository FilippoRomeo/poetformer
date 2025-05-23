import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. Load tokenizer & dataset
model_name = "mistralai/Mistral-7B-v0.1"
dataset = load_dataset("biglam/gutenberg-poetry-corpus", split="train[:1%]")  # small split for test

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(example["line"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 2. Load base model with 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# 3. Setup LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./models/lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# 5. Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 6. Train
trainer.train()

# 7. Save
trainer.save_model("./models/lora")
