# Poetformer

A generative-AI experiment comparing **base Mistral 7B** with a **LoRA fine-tuned poetry model** from the same prompt.

The project is less about building a general chatbot and more about observing what changes when a large language model is adapted towards a narrow literary corpus using parameter-efficient fine-tuning.

## Experiment

```text
prompt
  ├──► base Mistral 7B
  │        ↓
  │    base poem
  │
  └──► Mistral 7B + LoRA adapter
           ↓
     fine-tuned poem

          ↓
 side-by-side comparison
```

The comparison interface is built with Streamlit so both generations can be inspected from the same input.

## Stack

- `mistralai/Mistral-7B-v0.1`
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Streamlit
- public-domain poetry data from Project Gutenberg

## What the project explores

- parameter-efficient fine-tuning of a 7B language model
- stylistic adaptation from a specialised text corpus
- base-model vs adapted-model comparison
- prompt-controlled generation around themes and emotional direction
- separating the reusable base model from lightweight LoRA weights

## Quick start

Clone the repository:

```bash
git clone https://github.com/FilippoRomeo/poetformer.git
cd poetformer
```

Install the project dependencies with a Python environment appropriate for your hardware:

```bash
pip install -r requirements.txt
```

Run the comparison interface:

```bash
streamlit run compare_app.py
```

A CUDA-capable GPU is strongly preferable when loading and fine-tuning a model of this size.

## Fine-tuning

The LoRA training entry point is:

```bash
python training/fine_tune.py
```

The resulting adapter is stored separately under `models/lora/`, allowing the same base model to be used with or without the poetry adaptation.

## Example prompt

```text
Write a poem about dawn breaking over a quiet forest.
```

The Streamlit app sends the same prompt through the base and fine-tuned paths and presents the outputs side by side.

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

## Scope

This is a qualitative model-adaptation experiment, not a claim that the fine-tuned model is objectively "better" at poetry. The interesting output is the difference in language, structure, tone, and stylistic behaviour produced by the LoRA adaptation.

## References

- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Gutenberg Poetry Corpus](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT](https://github.com/huggingface/peft)

## License

Apache 2.0.
