from typing import List, Dict, Any
import yaml
import time


def load_model_configs(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["models"]


def batch_queries(queries: List[str], batch_size: int = 5) -> List[List[str]]:
    return [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]


def rate_limit_sleep(api_name: str):
    """Prevent hitting API rate limits"""
    limits = {"openai": 0.5, "anthropic": 1.0, "google": 2.0}
    time.sleep(limits.get(api_name.lower(), 1.0))
