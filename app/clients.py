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
from lime.lime_text import LimeTextExplainer
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


class ModelClient:
    """Base class for all API model clients"""

    _registry = {
        ModelNameEnum.GPT_3_5.value: "OpenAIClient",
        ModelNameEnum.CLAUDE_3_OPUS.value: "AnthropicClient",
        ModelNameEnum.GEMINI_PRO.value: "GeminiClient",
        ModelNameEnum.GROK_BETA.value: "xAIClient",
        ModelNameEnum.LLAMA_2_7B.value: "HuggingFaceModel",
        ModelNameEnum.MISTRAL_7B.value: "HuggingFaceModel",
        ModelNameEnum.DEEPSEEK_7B.value: "HuggingFaceModel",
        ModelNameEnum.QWEN_7B.value: "HuggingFaceModel",
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = settings.models[model_name]

    @classmethod
    def register_client(cls, name: str):
        """Decorator to register model client classes"""

        def wrapper(client_class):
            cls._registry[name] = client_class.__name__
            return client_class

        return wrapper

    @classmethod
    def get_client(cls, model_name: str, mitigation: str = None) -> "ModelClient":
        """Factory method to get the appropriate client"""
        if model_name not in cls._registry:
            raise ValueError(f"No client registered for model: {model_name}")
        client_class_name = cls._registry[model_name]
        client_class = globals().get(client_class_name)
        if not client_class:
            raise ValueError(f"Client class {client_class_name} not found")
        client = client_class(model_name)
        if mitigation == "rag":
            if not hasattr(client, "retriever"):
                client.retriever = PubMedRetriever()
        elif mitigation == "lora":
            pass
        return client

    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


@ModelClient.register_client("xai")
class xAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = self.config.api_key
        self.base_url = "https://api.x.ai/v1"

    async def generate(self, prompt, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": kwargs.get("model", "grok-beta"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            return response.json()["choices"][0]["message"]["content"]


@ModelClient.register_client("openai")
class OpenAIClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = openai.Client(api_key=self.config.api_key)
        self.retriever = PubMedRetriever() if self.config.enable_rag else None

    async def generate(self, prompt, **kwargs):
        if self.retriever and kwargs.get("use_rag", True):
            context = await self.retriever.retrieve(prompt, top_k=3)
            augmented_prompt = f"Medical Context:\n{context}\n\nQuestion: {prompt}"
        else:
            augmented_prompt = prompt
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4"),
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=kwargs.get("temperature", 0.3),
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
        return {"response": response.choices[0].message.content, "lime_explanation": exp.as_list()}


@ModelClient.register_client("anthropic")
class AnthropicClient(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic(api_key=self.config.api_key)

    async def generate(self, prompt, **kwargs):
        response = await self.client.messages.create(
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

    async def generate(self, prompt, **kwargs):
        response = await self.model.generate_content_async(prompt)
        return response.text


@ModelClient.register_client("huggingface")
class HuggingFaceModel(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() and settings.evaluation.use_gpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)

    async def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", settings.models[self.model_name].max_tokens),
            temperature=kwargs.get("temperature", settings.models[self.model_name].temperature),
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
