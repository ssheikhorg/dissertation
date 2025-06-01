from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any


class HuggingFaceModel:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.7),
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
