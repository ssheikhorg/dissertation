import openai
from anthropic import Anthropic
import google.generativeai as genai
from typing import Dict, Any
import httpx

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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


class ModelClient:
    """Base class for all API model clients"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class xAIClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = "https://api.x.ai/v1"  # From xAI docs

    def generate(self, prompt, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": kwargs.get("model", "grok-beta"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        with httpx.Client(headers=headers) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload)
            return response.json()["choices"][0]["message"]["content"]


class OpenAIClient(ModelClient):
    def __init__(self):
        super().__init__(config)
        self.client = openai.Client(api_key=config["api_key"])
        self.retriever = PubMedRetriever() if config.get("enable_rag", True) else None

    def generate(self, prompt, **kwargs):
        if self.retriever and kwargs.get("use_rag", True):
            # Retrieve relevant medical context
            context = self.retriever.retrieve(prompt, top_k=3)
            augmented_prompt = f"Medical Context:\n{context}\n\nQuestion: {prompt}"
        else:
            augmented_prompt = prompt

        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4"),
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=kwargs.get("temperature", 0.3),  # Lower temp for medical
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content


class AnthropicClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        self.client = Anthropic(api_key=config["api_key"])

    def generate(self, prompt, **kwargs):
        response = self.client.messages.create(
            model=kwargs.get("model", "claude-3-opus-20240229"),
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel("gemini-pro")

    def generate(self, prompt, **kwargs):
        response = self.model.generate_content(prompt)
        return response.text
