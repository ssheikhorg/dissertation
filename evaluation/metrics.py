from rouge_score import rouge_scorer


class MetricCalculator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'])

    def calculate(self, reference, prediction):
        return {
            'rougeL': self.rouge.score(reference, prediction)['rougeL'].fmeasure,
            'exact_match': float(reference.strip() == prediction.strip())
        }
