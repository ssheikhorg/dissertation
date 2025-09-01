import asyncio
import json
import os
import re
import string
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from pydantic import BaseModel

from .config import settings


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    publication_date: str
    journal: str


class DatasetLoader:
    def __init__(self):
        self.benchmarks = {
            "truthful_qa": {
                "path": "truthful_qa",
                "config": "generation",
                "split": "validation",
                "prompt_col": "question",
                "reference_col": "best_answer",
            },
            "winobias": {
                "path": "wino_bias",
                "split": "test",
                "prompt_col": "sentence",
                "reference_col": "target",
            },
            "pubmed_qa": {
                "path": "pubmed_qa",
                "config": "pqa_labeled",
                "split": "train",
                "prompt_col": "question",
                "reference_col": "long_answer",
            },
            "med_qa": {
                "path": "bigbio/med_qa",  # Fixed
                "config": "med_qa_en_source",  # From browse_page
                "split": "test",
                "prompt_col": "question",
                "reference_col": "answer",
            },
            "mimic_cxr": {
                "path": "itsanmolgupta/mimic-cxr-dataset",
                "split": "test",
                "prompt_col": "report",
                "reference_col": "label",
            },
        }
        self.diseases = {"cancer", "diabetes", "hypertension", "asthma", "arthritis"}
        self.drugs = {"aspirin", "ibuprofen", "metformin", "lisinopril", "atorvastatin"}
        self.anatomy = {"heart", "lung", "liver", "brain", "kidney"}
        try:
            loop = asyncio.get_running_loop()
            self._load_task = loop.create_task(self._async_load_medical_entities())
        except RuntimeError:
            # No running loop at import time; defer loading until ensure_loaded() is awaited
            self._load_task = None

    async def _async_load_medical_entities(self):
        """Asynchronously load medical entities in the background"""
        try:
            diseases = await load_disease_terms()
            drugs = load_drug_terms()
            anatomy = load_anatomy_terms()

            self.diseases = set(diseases)
            self.drugs = set(drugs)
            self.anatomy = set(anatomy)
        except Exception as e:
            print(f"Warning: Failed to load medical entities: {e}")

    async def ensure_loaded(self):
        """Wait for medical entities to be loaded if needed"""
        if self._load_task and not self._load_task.done():
            await self._load_task

    def load_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.benchmarks:
            raise ValueError(f"Unknown dataset: {name}")

        cfg = self.benchmarks[name]

        try:
            # Try loading with the configured parameters
            if cfg.get("config"):
                dataset = load_dataset(cfg["path"], cfg["config"], trust_remote_code=True)
            else:
                dataset = load_dataset(cfg["path"], trust_remote_code=True)

            # Handle different dataset structures
            if cfg["split"] in dataset:
                dataset_split = dataset[cfg["split"]]
            elif "train" in dataset:
                dataset_split = dataset["train"]  # Fallback to train split
            elif "validation" in dataset:
                dataset_split = dataset["validation"]  # Fallback to validation split
            elif "test" in dataset:
                dataset_split = dataset["test"]  # Fallback to test split
            else:
                # If no standard splits, use the first available split
                first_split = list(dataset.keys())[0]
                dataset_split = dataset[first_split]

            return dataset_split.to_pandas()

        except Exception as e:
            print(f"Error loading dataset {name}: {e}")

            # Special handling for med_qa which might have different structure
            if name == "med_qa":
                try:
                    print("Trying alternative med_qa loading approach...")
                    # Try loading without config
                    dataset = load_dataset("bigbio/med_qa", trust_remote_code=True)
                    if "test" in dataset:
                        return dataset["test"].to_pandas()
                    else:
                        return dataset[list(dataset.keys())[0]].to_pandas()
                except Exception as fallback_error:
                    print(f"Fallback med_qa loading also failed: {fallback_error}")

            # Fallback to pubmed_qa if the requested dataset fails
            print("Falling back to pubmed_qa dataset")
            fallback_cfg = self.benchmarks["pubmed_qa"]
            fallback_dataset = load_dataset(fallback_cfg["path"], fallback_cfg["config"], trust_remote_code=True)
            return fallback_dataset[fallback_cfg["split"]].to_pandas()

    def _get_fallback_prompts(self, n_samples: int) -> list[dict]:
        """Provide fallback medical prompts suitable for smaller models"""
        fallback_prompts = [
            {
                "question": "What are common diabetes symptoms?",
                "long_answer": "Common diabetes symptoms include increased thirst, frequent urination, and fatigue.",
            },
            {
                "question": "How does aspirin work?",
                "long_answer": "Aspirin reduces pain and inflammation by blocking certain enzymes.",
            },
            {
                "question": "What is hypertension?",
                "long_answer": "Hypertension is high blood pressure that can lead to health problems.",
            },
            {
                "question": "Liver function in human body?",
                "long_answer": "The liver detoxifies chemicals and produces bile for digestion.",
            },
        ]
        return fallback_prompts[:n_samples]

    def get_test_prompts(self, name: str, n_samples: int = 100) -> list[dict]:
        try:
            df = self.load_dataset(name)

            # Ensure we have enough samples
            n_samples = min(n_samples, len(df))
            samples = df.sample(n_samples) if len(df) > 0 else df

            # Handle different column name possibilities
            prompt_col = self.benchmarks[name]["prompt_col"]
            reference_col = self.benchmarks[name]["reference_col"]

            # Check if columns exist, use fallbacks if not
            available_columns = df.columns.tolist()
            if prompt_col not in available_columns:
                # Try to find a suitable prompt column
                for col in ["question", "input", "text", "sentence"]:
                    if col in available_columns:
                        prompt_col = col
                        break

            if reference_col not in available_columns:
                # Try to find a suitable reference column
                for col in ["answer", "output", "label", "target", "long_answer"]:
                    if col in available_columns:
                        reference_col = col
                        break

            # Return in a consistent format
            return [
                {
                    "prompt": str(row[prompt_col]) if pd.notna(row[prompt_col]) else "",
                    "reference": str(row[reference_col]) if pd.notna(row[reference_col]) else "",
                    "question": str(row[prompt_col]) if pd.notna(row[prompt_col]) else "",
                    "long_answer": str(row[reference_col]) if pd.notna(row[reference_col]) else "",
                }
                for _, row in samples.iterrows()
            ]

        except Exception as e:
            print(f"Error getting prompts from {name}: {e}")
            # Fallback to a simple set of medical questions
            return self._get_fallback_prompts(n_samples)

    async def load_medical_entities(self):
        """Load common medical entities to verify against"""
        try:
            self.diseases = set(await load_disease_terms())
            self.drugs = set(load_drug_terms())
            self.anatomy = set(load_anatomy_terms())
        except Exception as e:
            print(f"Warning: Failed to load medical entities: {e}")
            # Fallback to default terms
            self.diseases = {"cancer", "diabetes", "hypertension", "asthma", "arthritis"}
            self.drugs = {"aspirin", "ibuprofen", "metformin", "lisinopril", "atorvastatin"}
            self.anatomy = {"heart", "lung", "liver", "brain", "kidney"}


class PubMedRetriever:
    def __init__(self, api_key: str = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = api_key or settings.pubmed_api_key
        self.cache = {}  # Simple cache to avoid duplicate requests
        self.last_request_time = 0
        self.min_request_interval = 0.34  # Respect PubMed's rate limit (3 requests/second)

    async def _rate_limit(self):
        """Ensure we respect PubMed's rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    async def search(self, query: str, max_results: int = 3) -> list[PubMedArticle]:
        """Search PubMed for relevant articles"""
        cache_key = f"search:{query}:{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self._rate_limit()  # Make rate_limit async too
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

    async def fetch_articles(self, pmid_list: list[str]) -> list[PubMedArticle]:
        """Fetch full article details by PMID"""
        cache_key = f"articles:{','.join(pmid_list)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self._rate_limit()  # Make rate_limit async too
        params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient() as client:
            fetch_url = f"{self.base_url}/efetch.fcgi"
            response = await client.get(fetch_url, params=params)
            response.raise_for_status()

            articles = self._parse_pubmed_xml(response.text)
            self.cache[cache_key] = articles
            return articles

    def _parse_pubmed_xml(self, xml_content: str) -> list[PubMedArticle]:
        root = ET.fromstring(xml_content)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or "Unknown"
            title = article.findtext(".//ArticleTitle") or "No title"
            abstract = article.findtext(".//AbstractText") or ""  # Handle None case
            authors = [a.text for a in article.findall(".//Author/LastName") if a.text]
            journal = article.findtext(".//Journal/Title") or "Unknown journal"
            pub_date = article.findtext(".//PubDate/Year") or "Unknown date"

            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,  # Now guaranteed to be a string
                    authors=authors,
                    journal=journal,
                    publication_date=pub_date,
                )
            )
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


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text normalization"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def preprocess_batch(records: list[dict]) -> list[dict]:
        return [
            {
                "original_prompt": r.get("question", ""),  # Use 'question' as prompt
                "clean_prompt": TextPreprocessor.clean_text(r.get("question", "")),
                "original_reference": r.get("long_answer", r.get("reference", "")),  # Use 'long_answer' as reference
                "clean_reference": TextPreprocessor.clean_text(r.get("long_answer", r.get("reference", ""))),
            }
            for r in records
        ]


def get_dataset_stats(prompts: list[dict]) -> dict:
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


def load_test_prompts(
    dataset_name: str, n_samples: int, dataset_loader: DatasetLoader = DatasetLoader()
) -> list[dict]:
    try:
        records = dataset_loader.get_test_prompts(dataset_name, n_samples)
        return TextPreprocessor.preprocess_batch(records)
    except Exception as e:
        print(f"Warning: Failed to load prompts from {dataset_name}: {str(e)}")
        print("Using fallback prompts instead")
        # Use the dataset loader's fallback method
        fallback_prompts = dataset_loader._get_fallback_prompts(n_samples)
        return TextPreprocessor.preprocess_batch(fallback_prompts)


def load_baseline(model_name: str, dataset_name: str) -> dict:
    # Path to baseline results (create if doesn't exist)
    baseline_dir = os.path.join(settings.data_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_file = os.path.join(baseline_dir, f"{model_name}_{dataset_name}.json")

    try:
        if os.path.exists(baseline_file):
            with open(baseline_file) as f:
                return json.load(f)

        # Default baseline if no file exists
        return {
            "hallucination_rate": 0.25,  # Default 25% for unknown models
            "accuracy": 0.70,
            "fact_score": 0.75,
        }
    except Exception as e:
        raise ValueError(f"Failed to load baseline for {model_name}/{dataset_name}: {str(e)}")


def load_model_configs(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["models"]


def batch_queries(queries: list[str], batch_size: int = 5) -> list[list[str]]:
    return [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]


def rate_limit_sleep(api_name: str):
    """Prevent hitting API rate limits"""
    time.sleep(settings.datasets.rate_limits.get(api_name.lower(), 1.0))


async def load_disease_terms():
    retriever = PubMedRetriever()
    articles = await retriever.search("disease", max_results=10)  # Async call needs await
    return [article.title for article in articles]


def load_drug_terms():
    """Load drug terms from a hypothetical source."""
    try:
        with open("data/drugs.json") as f:
            return json.load(f)["drugs"]
    except FileNotFoundError:
        return ["aspirin", "ibuprofen", "metformin"]


def load_anatomy_terms():
    """Load anatomy terms from a hypothetical source."""
    try:
        with open("data/anatomy.json") as f:
            return json.load(f)["anatomy"]
    except FileNotFoundError:
        return ["heart", "lung", "liver"]
