from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict, List


class MetricCalculator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_all(self, references: List[str], predictions: List[str]) -> Dict:
        rouge_scores = [self.scorer.score(ref, pred)['rougeL'].fmeasure
                        for ref, pred in zip(references, predictions)]

        exact_matches = [float(ref.strip() == pred.strip())
                         for ref, pred in zip(references, predictions)]

        return {
            'rougeL': np.mean(rouge_scores),
            'exact_match': np.mean(exact_matches),
            'std_dev': np.std(rouge_scores)
        }