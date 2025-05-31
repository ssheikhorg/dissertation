from typing import Dict, List
import pandas as pd
from .metrics import MetricCalculator
from .bias_analysis import BiasAnalyzer


class ModelEvaluator:
    def __init__(self):
        self.metric_calc = MetricCalculator()
        self.bias_analyzer = BiasAnalyzer()

    def evaluate_model(self, model_name: str, prompts: List[Dict], model_client) -> pd.DataFrame:
        results = []
        for prompt in prompts:
            response = model_client.generate(prompt['clean_prompt'])

            metrics = self.metric_calc.calculate_all(
                [prompt['clean_reference']],
                [response]
            )

            bias = self.bias_analyzer.analyze_responses([response])

            results.append({
                'model': model_name,
                'prompt_id': prompt.get('id', hash(prompt['original_prompt'])),
                'prompt': prompt['original_prompt'],
                'reference': prompt['original_reference'],
                'response': response,
                **metrics,
                **bias
            })

        return pd.DataFrame(results)