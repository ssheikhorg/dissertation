from datasets import load_dataset
import pandas as pd
import re
import string
from typing import List, Dict, Any
import yaml
import time

from app.config import settings


def load_model_configs(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["models"]


def batch_queries(queries: List[str], batch_size: int = 5) -> List[List[str]]:
    return [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]


def rate_limit_sleep(api_name: str):
    """Prevent hitting API rate limits"""
    time.sleep(settings.datasets.rate_limits.get(api_name.lower(), 1.0))


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
                "path": "med_qa",
                "config": "en",
                "split": "test",
                "prompt_col": "question",
                "reference_col": "answer",
            },
            "mimic_cxr": {
                "path": "mimic_cxr",
                "split": "test",
                "prompt_col": "findings",
                "reference_col": "impression",
            },
        }

    def load_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.benchmarks:
            raise ValueError(f"Unknown dataset: {name}")

        cfg = self.benchmarks[name]
        dataset = load_dataset(cfg["path"], cfg.get("config"))[cfg["split"]]
        return dataset.to_pandas()

    def get_test_prompts(self, name: str, n_samples: int = 100) -> List[Dict]:
        df = self.load_dataset(name)
        samples = df.sample(min(n_samples, len(df)))
        return samples[
            [
                self.benchmarks[name]["prompt_col"],
                self.benchmarks[name]["reference_col"],
            ]
        ].to_dict("records")

    def load_medical_entities(self):
        """Load common medical entities to verify against"""
        self.diseases = set(load_disease_terms())
        self.drugs = set(load_drug_terms())
        self.anatomy = set(load_anatomy_terms())


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text normalization"""
        text = text.lower().strip()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def preprocess_batch(records: List[Dict]) -> List[Dict]:
        return [
            {
                "original_prompt": r["prompt"],
                "clean_prompt": TextPreprocessor.clean_text(r["prompt"]),
                "original_reference": r.get("reference", ""),
                "clean_reference": TextPreprocessor.clean_text(r.get("reference", "")),
            }
            for r in records
        ]
