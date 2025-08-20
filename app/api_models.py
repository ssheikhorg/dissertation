from typing import Any

import google.generativeai as genai
import httpx
import openai
import torch
from anthropic import Anthropic
from clients import ModelClient
from config import settings
from data import PubMedRetriever
from lime.lime_text import LimeTextExplainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class HuggingFaceModel:
    def __init__(self, config: dict[str, Any]):
        self.model_name = config["model_name"]
        self.device = "cuda" if torch.cuda.is_available() and settings.evaluation.use_gpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", settings.models[self.model_name].max_tokens),
            temperature=kwargs.get("temperature", settings.models[self.model_name].temperature),
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


@ModelClient.register_client("xai")
class xAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = self.config.api_key
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


@ModelClient.register_client("openai")
class OpenAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = openai.Client(api_key=self.config.api_key)
        self.retriever = PubMedRetriever() if self.config.enable_rag else None

    def generate(self, prompt, **kwargs):
        if self.retriever and kwargs.get("use_rag", True):
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
        explainer = LimeTextExplainer()
        exp = explainer.explain_instance(
            prompt,
            lambda x: [
                self.client.chat.completions.create(
                    model=kwargs.get("model", "gpt-4"),
                    messages=[{"role": "user", "content": p}],
                    temperature=0,
                )
                .choices[0]
                .message.content
                for p in x
            ],
            num_features=10,
        )
        return {"response": response, "lime_explanation": exp.as_list()}


@ModelClient.register_client("anthropic")
class AnthropicClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic(api_key=self.config.api_key)

    def generate(self, prompt, **kwargs):
        response = self.client.messages.create(
            model=kwargs.get("model", "claude-3-opus-20240229"),
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


@ModelClient.register_client("gemini")
class GeminiClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def generate(self, prompt, **kwargs):
        response = self.model.generate_content(prompt)
        return response.text
