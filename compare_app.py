import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ----- Model Paths and Loading -----
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "models/lora"

import os
print("Resolved LoRA path:", os.path.abspath(LORA_PATH))

@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True
    )

    # Load fine-tuned model
    lora = PeftModel.from_pretrained(base, LORA_PATH)
    lora.eval()

    return tokenizer, base, lora

# ----- Poem Generation -----
def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.9, top_p=0.95):
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

# ----- Streamlit UI -----
st.set_page_config(page_title="Poetformer Comparison", layout="centered")
st.title("🌟 Fine-Tuned vs Base Model Poetry Generator")

prompt = st.text_area("Enter your poem theme or feeling:", "Write a poem about the stars over a sleeping city")

if st.button("Generate and Compare"):
    with st.spinner("Loading models and generating poems..."):
        tokenizer, base_model, lora_model = load_models()

        prompt_input = f"### Instruction:\n{prompt}\n\n### Response:\n"

        base_poem = generate(base_model, tokenizer, prompt_input)
        lora_poem = generate(lora_model, tokenizer, prompt_input)

    st.subheader("🧠 Fine-Tuned Model Output")
    st.text_area("Poem from LoRA Model", value=lora_poem, height=300)

    st.subheader("😐 Base Model Output")
    st.text_area("Poem from Base Model", value=base_poem, height=300)

    st.success("Done! Compare and analyze the differences above.")

st.markdown("---")
st.markdown("Built with :blue_heart: using Mistral-7B and Streamlit.")
