from enum import Enum

import google.generativeai as genai
import httpx
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import torch
from anthropic import Anthropic
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings
from .data import PubMedRetriever, load_test_prompts
from .evaluators import ModelEvaluator, compare_models


class ModelNameEnum(Enum):
    GPT_3_5 = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-2"
    GEMINI_PRO = "gemini-2.0-flash-lite"
    GROK_BETA = "grok-beta"
    LLAMA_2_7B = "llama-2-7b"
    MISTRAL_7B = "mistral-7b"
    DEEPSEEK_7B = "deepseek-7b"
    QWEN_7B = "qwen-7b"


class RAGEnhancedClient:
    def __init__(self, base_client):
        self.base_client = base_client
        self.retriever = PubMedRetriever()

    async def generate(self, prompt, **kwargs):
        # Retrieve relevant medical context
        context = await self.retriever.retrieve(prompt, top_k=3)
        augmented_prompt = f"Medical Context:\n{context}\n\nQuestion: {prompt}"

        # Generate response with context
        return await self.base_client.generate(augmented_prompt, **kwargs)


class LoRAEnhancedClient:
    def __init__(self, base_client):
        self.base_client = base_client
        # Load fine-tuned LoRA weights would go here

    async def generate(self, prompt, **kwargs):
        # Apply LoRA fine-tuning logic
        # This would involve loading specific weights and modifying the generation
        return await self.base_client.generate(prompt, **kwargs)


class EnsembleClient:
    def __init__(self, base_client):
        self.base_client = base_client
        self.models = [base_client]  # Would include multiple models in real implementation

    async def generate(self, prompt, **kwargs):
        # Generate responses from multiple models and ensemble them
        responses = []
        for model in self.models:
            response = await model.generate(prompt, **kwargs)
            responses.append(response)

        # Simple majority voting or more sophisticated ensembling
        return self._ensemble_responses(responses)

    def _ensemble_responses(self, responses):
        # Basic ensemble logic - would be more sophisticated in production
        return responses[0]  # Placeholder


class ModelClient:
    """Base class for all API model clients"""

    _registry = {
        "gpt-3.5-turbo": "OpenAIClient",
        "gpt-4": "OpenAIClient",
        "claude-2": "AnthropicClient",
        "gemini-2.0-flash-lite": "GeminiClient",
        "grok-beta": "xAIClient",
        "llama-2-7b": "HuggingFaceModel",
        "mistral-7b": "HuggingFaceModel",
        "deepseek-7b": "HuggingFaceModel",
        "qwen-7b": "HuggingFaceModel",
    }

    @classmethod
    def register_client(cls, name: str):
        """Decorator to register model client classes"""

        def wrapper(client_class):
            cls._registry[name] = client_class.__name__
            return client_class

        return wrapper

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Safe access to models config with defaults
        self.config = getattr(settings, "models", {}).get(model_name, {})
        self.provider = self.config.get("provider", "huggingface")

    @classmethod
    def get_client(cls, model_name: str, mitigation: str = None) -> "ModelClient":
        """Factory method to get the appropriate client with mitigation"""
        if model_name not in cls._registry:
            # Fallback to a known model but keep the original name for tracking
            fallback_model = "llama-2-7b"
            print(f"Warning: Model {model_name} not found, using {fallback_model} as fallback")
            model_name = fallback_model

        client_class_name = cls._registry[model_name]
        client_class = globals().get(client_class_name)

        if not client_class:
            # Fallback to HuggingFace model
            from .clients import HuggingFaceModel

            client_class = HuggingFaceModel

        # Create client instance
        client = client_class(model_name)

        # Apply mitigation strategies
        if mitigation == "rag":
            client = RAGEnhancedClient(client)
        elif mitigation == "lora":
            client = LoRAEnhancedClient(client)
        elif mitigation == "ensemble":
            client = EnsembleClient(client)

        return client

    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


@ModelClient.register_client("xai")
class xAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Use config API key first, then fallback to global setting
        self.api_key = self.config.get("api_key") or getattr(settings, "xai_api_key", None)
        self.base_url = "https://api.x.ai/v1"

    async def generate(self, prompt, **kwargs):
        if not self.api_key:
            return "xAI API key not configured. Please set MODEL_XAI_API_KEY in your .env file."

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": kwargs.get("model", "grok-beta"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
        }
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            return response.json()["choices"][0]["message"]["content"]


# Update all client classes to use safe config access
@ModelClient.register_client("openai")
class OpenAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Use config API key first, then fallback to global setting
        api_key = self.config.get("api_key") or getattr(settings, "openai_api_key", None)
        self.client = openai.Client(api_key=api_key)
        enable_rag = self.config.get("enable_rag", getattr(settings, "enable_rag", True))
        self.retriever = PubMedRetriever() if enable_rag else None

    async def generate(self, prompt, **kwargs):
        if self.retriever and kwargs.get("use_rag", True):
            context = await self.retriever.retrieve(prompt, top_k=3)
            augmented_prompt = f"Medical Context:\n{context}\n\nQuestion: {prompt}"
        else:
            augmented_prompt = prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=kwargs.get(
                "temperature", self.config.get("temperature", getattr(settings, "temperature", 0.3))
            ),
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", getattr(settings, "max_tokens", 1000))),
        )
        return response.choices[0].message.content


@ModelClient.register_client("anthropic")
class AnthropicClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Use config API key first, then fallback to global setting
        api_key = self.config.get("api_key") or getattr(settings, "anthropic_api_key", None)
        self.client = Anthropic(api_key=api_key)

    async def generate(self, prompt, **kwargs):
        response = await self.client.messages.create(
            model=kwargs.get("model", "claude-3-opus-20240229"),
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


@ModelClient.register_client("gemini")
class GeminiClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    async def generate(self, prompt, **kwargs):
        response = await self.model.generate_content_async(prompt)
        return response.text


@ModelClient.register_client("huggingface")
class HuggingFaceModel(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() and getattr(settings, "use_gpu", False) else "cpu"

        torch_dtype = torch.float32 if self.device == "mps" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device != "mps" else None,
        ).to(self.device)

    async def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get(
                "max_tokens", self.config.get("max_tokens", getattr(settings, "max_tokens", 1000))
            ),
            temperature=kwargs.get(
                "temperature", self.config.get("temperature", getattr(settings, "temperature", 0.3))
            ),
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


async def generate_visualization_data(
    model_names: list[str] = None,
    datasets: list[str] = ["pubmed_qa", "med_qa"],
    metrics: list[str] = ["accuracy", "hallucination_rate", "toxicity_score"],
    bias_types: list[str] = ["Gender", "Racial", "Political"],
    n_samples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate DataFrames for visualization from model evaluation results.

    Args:
        model_names: List of model names to evaluate (default: all registered models).
        datasets: List of dataset names to evaluate on.
        metrics: List of metrics to include in the general DataFrame.
        bias_types: List of bias types for the bias DataFrame.
        n_samples: Number of samples per dataset.

    Returns:
        Tuple of (general_df, bias_df):
        - general_df: DataFrame for create_grouped_barplot/metric_comparison_bars.
        - bias_df: DataFrame for stacked_bias_bars.
    """
    if model_names is None:
        model_names = list(ModelClient._registry.keys())

    general_data = []
    bias_data = []

    for dataset in datasets:
        prompts = load_test_prompts(dataset_name=dataset, n_samples=n_samples)
        results = await compare_models(model_names, prompts)  # Async call
        for model, result in results["models"].items():
            if result.get("metrics"):
                general_data.append({"model": model, "dataset": dataset, **result["metrics"]})
                # Generate bias data (placeholder: assumes bias_analyzer results)
                bias_scores = await ModelEvaluator().evaluate_model(model, prompts, mitigation=None)
                for bias_type in bias_types:
                    bias_data.append(
                        {
                            "model": model,
                            "bias_type": bias_type,
                            "score": bias_scores.get("toxicity_score", 0.0),  # Example: use real bias scores
                        }
                    )

    general_df = pd.DataFrame(general_data)
    bias_df = pd.DataFrame(bias_data)

    return general_df, bias_df


def create_grouped_barplot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    hue_column: str,
    title: str = "Model Performance Comparison",
    x_label: str = "",
    y_label: str = "Score",
    palette: str = "muted",
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=data,
        x=x_column,
        y=y_column,
        hue=hue_column,
        palette=palette,
        errorbar=None,
        estimator=np.mean,
    )
    plt.title(title, pad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def metric_comparison_bars(
    results_df: pd.DataFrame,
    metrics: list[str],
    model_col: str = "model",
    title: str = "Model Performance Metrics Comparison",
    save_path: str | None = None,
) -> None:
    melted_data = results_df.melt(id_vars=[model_col], value_vars=metrics, var_name="metric", value_name="score")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=melted_data,
        x="metric",
        y="score",
        hue=model_col,
        palette="Set2",
        errorbar=None,
    )
    plt.title(title, pad=20)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def stacked_bias_bars(
    results_df: pd.DataFrame,
    bias_types: list[str],
    model_col: str = "model",
    title: str = "Bias Composition Across Models",
    save_path: str | None = None,
) -> None:
    bias_totals = results_df.groupby(model_col)[bias_types].sum()
    bias_proportions = bias_totals.div(bias_totals.sum(axis=1), axis=0)
    ax = bias_proportions.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="RdYlGn_r", title=title)
    plt.title(title, pad=20)
    plt.xlabel("Model")
    plt.ylabel("Proportion of Bias")
    plt.xticks(rotation=45)
    plt.legend(title="Bias Types", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_heatmap(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    value_column: str,
    title: str = "Model Performance Heatmap",
    cmap: str = "YlOrRd",
    figsize: tuple = (12, 8),
    annot: bool = True,
    normalize: bool = False,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=figsize)
    heatmap_data = data.pivot_table(index=y_axis, columns=x_axis, values=value_column, aggfunc=np.mean)
    if normalize:
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": value_column},
    )
    plt.title(title, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def bias_heatmap(
    results_df: pd.DataFrame,
    model_names: list[str],
    bias_types: list[str],
    save_path: str | None = None,
) -> None:
    bias_data = results_df[results_df["model"].isin(model_names)]
    plt.figure(figsize=(12, 8))
    heatmap_data = bias_data.groupby(["model", "bias_type"])["score"].mean().unstack()
    heatmap_data = heatmap_data.reindex(columns=bias_types)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        linewidths=0.5,
        cbar_kws={"label": "Bias Score (higher = more biased)"},
    )
    plt.title("Bias Analysis Across Models", pad=20)
    plt.xlabel("Bias Type")
    plt.ylabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_radar_plot(results: dict[str, dict], metrics: list[str]) -> None:
    import plotly.express as px  # Moved here to avoid top-level import

    plot_data = []
    for model, scores in results.items():
        for metric in metrics:
            plot_data.append({"Model": model, "Metric": metric, "Score": scores[metric]})
    df = pd.DataFrame(plot_data)
    fig = px.line_polar(
        df,
        r="Score",
        theta="Metric",
        color="Model",
        line_close=True,
        template="plotly_dark",
        title="Model Performance Comparison",
    )
    fig.show()


class ReportGenerator:
    def __init__(self, template_dir="templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_html_report(self, results: dict, output_path: str) -> None:
        df = pd.DataFrame.from_dict(results, orient="index")
        template = self.env.get_template("report_template.html")
        html_content = template.render(models_results=df.to_html(classes="data"), metrics=list(df.columns))
        with open(output_path, "w") as f:
            f.write(html_content)


def plot_hallucination_reduction(results: dict, save_path: str = None):
    df = pd.DataFrame.from_dict(results, orient="index")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df.reset_index(), x="index", y="hallucination_rate", hue="mitigation")
    plt.title("Hallucination Rate Reduction by Mitigation Strategy")
    plt.xlabel("Model")
    plt.ylabel("Hallucination Rate (%)")
    plt.xticks(rotation=45)
    plt.axhline(y=20, color="r", linestyle="--", label="Target Threshold")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
