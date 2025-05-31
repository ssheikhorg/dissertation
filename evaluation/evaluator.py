from typing import Dict, List
import pandas as pd
from .metrics import MetricCalculator
from .bias_analysis import BiasAnalyzer
from .hallucination import HallucinationEvaluator
from .consistency import ConsistencyEvaluator


class ModelEvaluator:
    def __init__(self, config: Dict):
        """
        Initialize evaluation components.

        Args:
            config: Evaluation configuration dictionary
        """
        self.metric_calc = MetricCalculator(config)
        self.bias_analyzer = BiasAnalyzer(config)
        self.hallucination_eval = HallucinationEvaluator(config)
        self.consistency_eval = ConsistencyEvaluator(config)
        self.config = config

    def evaluate_model(self, model_name: str, prompts: List[Dict], model_client) -> pd.DataFrame:
        """
        Comprehensive model evaluation.

        Args:
            model_name: Name of model being evaluated
            prompts: List of prompt dictionaries
            model_client: Model client instance

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for prompt in prompts:
            # Get model response
            response = model_client.generate(prompt['clean_prompt'])

            # Basic metrics
            metrics = self.metric_calc.calculate_all(
                [prompt['clean_reference']],
                [response]
            )

            # Bias analysis
            bias = self.bias_analyzer.analyze_responses([response])

            # Hallucination detection
            hallucination = self.hallucination_eval.factual_consistency(
                prompt['clean_reference'],
                response
            )

            # Consistency evaluation (requires multiple responses)
            consistency = {'response_consistency': None}
            if 'variations' in prompt:  # If we have multiple versions of the prompt
                variations_responses = [
                    model_client.generate(v) for v in prompt['variations']
                ]
                consistency = self.consistency_eval.response_consistency(
                    [response] + variations_responses
                )

            # Self-consistency
            self_consistency = self.consistency_eval.self_consistency(response)

            results.append({
                'model': model_name,
                'prompt_id': prompt.get('id', hash(prompt['original_prompt'])),
                'prompt': prompt['original_prompt'],
                'reference': prompt['original_reference'],
                'response': response,
                **metrics,
                **bias,
                **hallucination,
                **consistency,
                **self_consistency
            })

        return pd.DataFrame(results)