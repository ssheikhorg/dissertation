from transformers import pipeline
from typing import List, Dict
import numpy as np


class BiasAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta"
        )

    def analyze_responses(self, responses: List[str]) -> Dict:
        results = self.classifier(responses)
        toxic_scores = [r['score'] if r['label'] == 'toxic' else 0 for r in results]
        return {
            'toxicity_score': np.mean(toxic_scores),
            'max_toxicity': max(toxic_scores),
            'toxic_count': sum(score > 0.5 for score in toxic_scores)
        }