# Poetformer

Fine-tuned poetry generation with Mistral 7B and LoRA.

Poetformer is a poetry-generation experiment that compares output from:

- the base `mistralai/Mistral-7B-v0.1` model;
- a LoRA fine-tuned version trained on public-domain poetry from Project Gutenberg.

The project uses Streamlit, Hugging Face Transformers, PEFT, and PyTorch.

## Features

- Compare base and fine-tuned model output from the same prompt
- Instruction-style prompting for stylistic control
- Theme and emotion-driven generation
- Lightweight LoRA adapter training

## Quick start

### Clone

```bash
git clone https://github.com/FilippoRomeo/poetformer.git
cd poetformer
```

### Install dependencies

Use Python 3.10+ with a CUDA-compatible PyTorch installation when running on GPU:

```bash
pip install -r requirements.txt
```

### Run the comparison app

```bash
streamlit run compare_app.py
```

## Project structure

```text
poetformer/
├── app/
├── data/
├── generate/
├── models/
├── training/
├── compare_app.py
└── README.md
```

## Fine-tuning

PEFT is used to fine-tune the base model with LoRA:

```bash
python training/fine_tune.py
```

The adapter is saved under `models/lora/`.

## Example prompt

```text
Write a poem about dawn breaking over a quiet forest.
```

The comparison app shows the base-model and LoRA outputs side by side.

## Credits

- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Gutenberg Poetry Corpus](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [Streamlit](https://streamlit.io/)

## License

Apache 2.0.
