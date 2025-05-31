from typing import List, Dict
import pandas as pd
from app.models.model_client import ModelClient
from app.utils.dataset_loader import DatasetLoader
from app.utils.preprocessor import TextPreprocessor
from evaluation.bias_analysis import BiasAnalyzer
from evaluation.consistency import ConsistencyEvaluator
from evaluation.metrics import MetricCalculator


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
            response = model_client.generate(prompt['clean_prompt'])

            metrics = self.metric_calc.calculate_all(
                [prompt['clean_reference']],
                [response]
            )

            bias = self.bias_analyzer.analyze_responses([response])
            self_consistency = self.consistency_eval.self_consistency(response)

            results.append({
                'model': model_name,
                'prompt': prompt['original_prompt'],
                'response': response,
                **metrics,
                **bias,
                **self_consistency
            })

        return pd.DataFrame(results)

    def compare_models(self, model_names: List[str], prompts: List[Dict]) -> pd.DataFrame:
        all_results = []
        for model_name in model_names:
            results = self.evaluate_model(model_name, prompts)
            all_results.append(results)
        return pd.concat(all_results)
