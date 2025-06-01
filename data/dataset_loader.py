from datasets import load_dataset
from typing import Dict, List, Union
import pandas as pd


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
