from typing import List, Dict
import pandas as pd
from pydantic import BaseModel

from app.api_models import (
    HuggingFaceModel,
    xAIClient,
    GeminiClient,
    AnthropicClient,
    OpenAIClient,
)
from app.config import settings
from app.model_evaluator import MetricCalculator, BiasAnalyzer, ConsistencyEvaluator
from app.retrievers import PubMedRetriever


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    journal: str


class ModelEvaluator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metric_calc = MetricCalculator()
        self.bias_analyzer = BiasAnalyzer()
        self.consistency_eval = ConsistencyEvaluator(self.config)

    def evaluate_model(self, model_name: str, prompts: List[Dict]) -> pd.DataFrame:
        model_client = ModelClient.get_client(model_name)
        results = []

        for prompt in prompts:
            response = model_client.generate(prompt["clean_prompt"])

            metrics = self.metric_calc.calculate_all(
                [prompt["clean_reference"]], [response]
            )

            bias = self.bias_analyzer.analyze_responses([response])
            self_consistency = self.consistency_eval.self_consistency(response)

            results.append(
                {
                    "model": model_name,
                    "prompt": prompt["original_prompt"],
                    "response": response,
                    **metrics,
                    **bias,
                    **self_consistency,
                }
            )

        return pd.DataFrame(results)

    def compare_models(
        self, model_names: List[str], prompts: List[Dict]
    ) -> pd.DataFrame:
        all_results = []
        for model_name in model_names:
            results = self.evaluate_model(model_name, prompts)
            all_results.append(results)
        return pd.concat(all_results)


class ModelClient:
    """Base class for all API model clients"""

    _registry = {
        "gpt-4": OpenAIClient,
        "claude-3-opus-20240229": AnthropicClient,
        "gemini-pro": GeminiClient,
        "grok-beta": xAIClient,
        "llama-2-7b": HuggingFaceModel,  # Add for LLaMA, Mistral, DeepSeek, Qwen
        "mistral-7b": HuggingFaceModel,
        "deepseek-7b": HuggingFaceModel,
        "qwen-7b": HuggingFaceModel,
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = settings.models[model_name]

    @classmethod
    def register_client(cls, name: str):
        """Decorator to register model client classes"""

        def wrapper(client_class):
            cls._registry[name] = client_class
            return client_class

        return wrapper

    @classmethod
    def get_client(cls, model_name: str, mitigation: str = None):
        """Factory method to get the appropriate client"""
        if model_name not in cls._registry:
            raise ValueError(f"No client registered for model: {model_name}")

        client = cls._registry[model_name](model_name)

        if mitigation == "rag":
            if not hasattr(client, "retriever"):
                client.retriever = PubMedRetriever()
        elif mitigation == "lora":
            # Initialize LoRA adapter if needed
            pass

        return client

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
