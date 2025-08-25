# Poetformer
# 🌟 Poetformer: Fine-Tuned Poetry Generation with Mistral 7B

**Poetformer** is a poetry generation app that compares outputs from:
- The **base** `mistralai/Mistral-7B-v0.1` model
- A **LoRA fine-tuned** version trained on public-domain poetry (Gutenberg corpus)

Built with ❤️ using Streamlit, Hugging Face Transformers, PEFT, and PyTorch.

---

## 🌍 Demo

Run locally with:
```bash
streamlit run compare_app.py
```

---

## 🔧 Features

- Compare fine-tuned vs base model poetry output on the same prompt
- Built-in instruction-style prompt format for stylistic control
- Emotion- and theme-driven input (e.g. "loneliness", "first snow", "melancholy")
- Lightweight LoRA adapter (trainable in 1 GPU session)

---

## 🚀 Quickstart

### 1. Clone Repo
```bash
git clone https://github.com/yourusername/poetformer.git
cd poetformer
```

### 2. Install Dependencies
Make sure you're using a Python 3.10+ Conda environment with CUDA-compatible PyTorch:
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run compare_app.py
```

---

## 🌐 Project Structure

```
poetformer/
├── app/                     # Gradio or Streamlit apps
├── data/                    # Raw and processed poetry datasets
├── generate/                # Inference scripts
├── models/                  # LoRA fine-tuned model directory
├── training/                # Fine-tuning pipeline
├── compare_app.py          # Streamlit app
└── README.md
```

---

## 🔄 Fine-Tuning (LoRA)

We use [PEFT](https://github.com/huggingface/peft) to fine-tune the base model on poetic outputs:
```bash
python training/fine_tune.py
```

Model is saved to `models/lora/`.

---

## 🌈 Example Prompt

Enter a freeform theme, emotion, or instruction:
```
Write a poem about dawn breaking over a quiet forest.
```

Outputs:
- ✨ LoRA: Poetic, stanza-based, verse-formatted
- 🧱 Base: More generic or prose-like

---

## 🤝 Contributing

Pull requests welcome! Ideas:
- Add rhyme detection/metrics
- Integrate feedback buttons
- Train on emotional poetry corpora

---

## 🎓 Credits

- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) by MistralAI
- [Gutenberg Poetry Dataset](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Streamlit](https://streamlit.io/)

---

## 📅 License

Apache 2.0. Free to use for research and remix.
