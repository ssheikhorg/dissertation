import torch
from transformers import pipeline


class HuggingFaceModel:
    def __init__(self, model_name):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]['generated_text']
