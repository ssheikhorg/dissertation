from concurrent.futures import ThreadPoolExecutor

import httpx
from typing import List
from app.config import settings
import time
import numpy as np

from typing import Dict
import pandas as pd
from config import settings
from .dataset_loader import DatasetLoader
from .evaluations import ModelEvaluator
import json
import os

from .models import PubMedArticle

# Initialize dataset loader
dataset_loader = DatasetLoader()
model_evaluator = ModelEvaluator()


def compare_models(model_names: List[str], prompts: List[Dict]) -> Dict:
    """
    Compare multiple models on the same set of prompts

    Args:
        model_names: List of model names to compare
        prompts: List of preprocessed prompts to evaluate on

    Returns:
        Dictionary containing comparison results with:
        - Individual model results
        - Comparative analysis
        - Statistical significance
    """
    # Validate input
    if not model_names:
        raise ValueError("At least one model name must be provided")

    if not prompts:
        raise ValueError("No prompts provided for evaluation")

    # Evaluate each model (parallelized for efficiency)
    with ThreadPoolExecutor() as executor:
        futures = {
            model_name: executor.submit(evaluate_single_model, model_name, prompts)
            for model_name in model_names
        }
        all_results = {
            model_name: future.result() for model_name, future in futures.items()
        }

    # Generate comparative analysis
    comparison = generate_comparison_metrics(all_results)

    return {
        "models": all_results,
        "comparison": comparison,
        "prompt_count": len(prompts),
        "dataset_stats": get_dataset_stats(prompts),
    }


def evaluate_single_model(model_name: str, prompts: List[Dict]) -> Dict:
    """Evaluate a single model on the given prompts"""
    try:
        # Get per-prompt results
        results_df = model_evaluator.evaluate_model(model_name, prompts)

        # Calculate aggregate metrics
        return {
            "metrics": calculate_aggregate_metrics(results_df),
            "sample_responses": get_sample_responses(results_df),
            "error_analysis": analyze_errors(results_df),
        }
    except Exception as e:
        return {"error": str(e), "metrics": None, "sample_responses": None}


def calculate_aggregate_metrics(results_df: pd.DataFrame) -> Dict:
    """Calculate aggregate metrics from individual results"""
    if results_df.empty:
        return {}

    return {
        "accuracy": results_df["exact_match"].mean(),
        "hallucination_rate": 1 - results_df["fact_score"].mean(),
        "toxicity_score": results_df["toxicity_score"].mean(),
        "consistency": results_df["self_consistency"].mean(),
        "rougeL": results_df["rougeL"].mean(),
        "medical_bleu": results_df.get("medical_bleu", np.nan).mean(),
        "response_length": results_df["response"].str.len().mean(),
    }


def get_sample_responses(results_df: pd.DataFrame, n: int = 3) -> List[Dict]:
    """Get sample responses with reference comparisons"""
    samples = []
    for _, row in results_df.sample(min(n, len(results_df))).iterrows():
        samples.append(
            {
                "prompt": row["prompt"],
                "reference": row["reference"],
                "response": row["response"],
                "fact_score": row["fact_score"],
                "is_contradiction": row.get("is_contradiction", False),
            }
        )
    return samples


def analyze_errors(results_df: pd.DataFrame) -> Dict:
    """Analyze common error patterns"""
    if results_df.empty:
        return {}

    # Get examples where fact_score < 0.5 (likely hallucinations)
    hallucinations = results_df[results_df["fact_score"] < 0.5]

    return {
        "hallucination_examples": get_sample_responses(hallucinations, 2)
        if not hallucinations.empty
        else [],
        "common_error_patterns": find_common_patterns(hallucinations["response"]),
        "hallucination_rate": len(hallucinations) / len(results_df),
    }


def find_common_patterns(responses: pd.Series) -> List[str]:
    """Identify common patterns in erroneous responses"""
    # This is a placeholder - implement proper NLP analysis
    from collections import Counter
    from nltk import ngrams

    # Simple n-gram analysis
    all_ngrams = []
    for text in responses:
        tokens = text.lower().split()
        all_ngrams.extend(ngrams(tokens, 3))

    return [" ".join(gram) for gram, count in Counter(all_ngrams).most_common(3)]


def generate_comparison_metrics(all_results: Dict[str, Dict]) -> Dict:
    """Generate comparative metrics between models"""
    comparison = {}
    metrics = ["accuracy", "hallucination_rate", "consistency", "rougeL"]

    for metric in metrics:
        values = {
            model: results["metrics"][metric]
            for model, results in all_results.items()
            if results.get("metrics")
        }

        if values:
            comparison[metric] = {
                "best_model": max(values.items(), key=lambda x: x[1])[0],
                "worst_model": min(values.items(), key=lambda x: x[1])[0],
                "range": max(values.values()) - min(values.values()),
                "mean": np.mean(list(values.values())),
                "values": values,
            }

    return comparison


def get_dataset_stats(prompts: List[Dict]) -> Dict:
    """Calculate basic statistics about the evaluation dataset"""
    ref_lengths = [len(p["original_reference"]) for p in prompts]
    prompt_lengths = [len(p["original_prompt"]) for p in prompts]

    return {
        "prompt_length": {
            "mean": np.mean(prompt_lengths),
            "max": max(prompt_lengths),
            "min": min(prompt_lengths),
        },
        "reference_length": {
            "mean": np.mean(ref_lengths),
            "max": max(ref_lengths),
            "min": min(ref_lengths),
        },
        "total_prompts": len(prompts),
    }


def load_test_prompts(dataset_name: str, n_samples: int) -> List[Dict]:
    """
    Load test prompts from the specified dataset

    Args:
        dataset_name: Name of the dataset to load
        n_samples: Number of samples to load

    Returns:
        List of prompt dictionaries with original/clean versions
    """
    try:
        records = dataset_loader.get_test_prompts(dataset_name, n_samples)
        return dataset_loader.TextPreprocessor.preprocess_batch(records)
    except Exception as e:
        raise ValueError(f"Failed to load prompts from {dataset_name}: {str(e)}")


def load_baseline(model_name: str, dataset_name: str) -> Dict:
    """
    Load baseline hallucination rates for a model/dataset combination

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Dictionary containing baseline metrics
    """
    # Path to baseline results (create if doesn't exist)
    baseline_dir = os.path.join(settings.data_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_file = os.path.join(baseline_dir, f"{model_name}_{dataset_name}.json")

    try:
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                return json.load(f)

        # Default baseline if no file exists
        return {
            "hallucination_rate": 0.25,  # Default 25% for unknown models
            "accuracy": 0.70,
            "fact_score": 0.75,
        }
    except Exception as e:
        raise ValueError(
            f"Failed to load baseline for {model_name}/{dataset_name}: {str(e)}"
        )


class MedicalModelEvaluator:
    def __init__(self):
        """
        Initialize medical-specific evaluator with healthcare-focused metrics
        """
        from .evaluations import (
            HallucinationEvaluator,
            MetricCalculator,
            ConsistencyEvaluator,
        )

        self.hallucination_eval = HallucinationEvaluator()
        self.metric_calc = MetricCalculator()
        self.consistency_eval = ConsistencyEvaluator(settings.evaluation)

    def evaluate(self, model_client, prompts: List[Dict]) -> Dict:
        """
        Evaluate model performance on medical prompts

        Args:
            model_client: Initialized model client
            prompts: List of preprocessed prompts

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        results = {
            "total_prompts": len(prompts),
            "hallucination_rate": 0,
            "accuracy": 0,
            "fact_score": 0,
            "medical_bleu": 0,
            "toxicity_score": 0,
            "fabrication_score": 0,
            "guideline_violations": 0,
        }

        # Aggregate metrics across all prompts
        all_metrics = []

        for prompt in prompts:
            response = model_client.generate(prompt["clean_prompt"])

            # Basic text metrics
            metrics = self.metric_calc.calculate_all(
                [prompt["clean_reference"]], [response]
            )

            # Medical-specific evaluations
            hallucination = self.hallucination_eval.factual_consistency(
                prompt["clean_reference"], response
            )

            fabrication = self.hallucination_eval.detect_medical_fabrications(response)
            guidelines = self.hallucination_eval.check_clinical_guidelines(response)

            # Aggregate results
            all_metrics.append(
                {
                    **metrics,
                    **hallucination,
                    **fabrication,
                    **guidelines,
                    "response": response,
                }
            )

        # Calculate aggregate scores
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            results.update(
                {
                    "hallucination_rate": 1 - df["fact_score"].mean(),
                    "accuracy": df["exact_match"].mean(),
                    "fact_score": df["fact_score"].mean(),
                    "medical_bleu": df.get("medical_bleu", 0).mean(),
                    "toxicity_score": df.get("toxicity_score", 0).mean(),
                    "fabrication_score": df.get("fabrication_score", 0).mean(),
                    "guideline_violations": df.get("guideline_violations", 0).mean(),
                    "sample_responses": df["response"]
                    .iloc[:3]
                    .tolist(),  # Include sample responses
                }
            )

        return results


class PubMedRetriever:
    def __init__(self, api_key: str = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = api_key or settings.pubmed_api_key
        self.cache = {}  # Simple cache to avoid duplicate requests
        self.last_request_time = 0
        self.min_request_interval = (
            0.34  # Respect PubMed's rate limit (3 requests/second)
        )

    def _rate_limit(self):
        """Ensure we respect PubMed's rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    async def search(self, query: str, max_results: int = 3) -> List[PubMedArticle]:
        """Search PubMed for relevant articles"""
        cache_key = f"search:{query}:{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient() as client:
            search_url = f"{self.base_url}/esearch.fcgi"
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            article_ids = data.get("esearchresult", {}).get("idlist", [])
            if not article_ids:
                return []

            articles = await self.fetch_articles(article_ids)
            self.cache[cache_key] = articles
            return articles

    async def fetch_articles(self, pmid_list: List[str]) -> List[PubMedArticle]:
        """Fetch full article details by PMID"""
        cache_key = f"articles:{','.join(pmid_list)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient() as client:
            fetch_url = f"{self.base_url}/efetch.fcgi"
            response = await client.get(fetch_url, params=params)
            response.raise_for_status()

            # Parse XML response (simplified - in practice you'd use lxml or similar)
            articles = self._parse_pubmed_xml(response.text)
            self.cache[cache_key] = articles
            return articles

    def _parse_pubmed_xml(self, xml_content: str) -> List[PubMedArticle]:
        """Simplified XML parser for PubMed results"""
        # In a real implementation, you'd use lxml or similar for proper parsing
        # This is a simplified version that would need proper XML parsing
        articles = []
        # Example parsing logic (would need proper XML parsing):
        # Split by <PubmedArticle> tags and extract relevant fields
        return articles

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant medical context for a query"""
        articles = await self.search(query, max_results=top_k)
        if not articles:
            return "No relevant medical context found."

        context = []
        for article in articles:
            context.append(f"Title: {article.title}")
            if article.abstract:
                context.append(f"Abstract: {article.abstract}")
            context.append(f"Journal: {article.journal}, {article.publication_date}")
            context.append("---")

        return "\n".join(context)[:5000]  # Limit context length
