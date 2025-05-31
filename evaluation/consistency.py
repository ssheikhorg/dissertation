from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import defaultdict


class ConsistencyEvaluator:
    def __init__(self, config: Dict):
        """
        Initialize consistency evaluation tools.

        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.similarity_model = SentenceTransformer(
            'all-mpnet-base-v2',
            device='cuda' if config.get("use_gpu", True) else 'cpu'
        )

        self.entailment_model = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=0 if config.get("use_gpu", True) else -1
        )

    def response_consistency(self, responses: List[str]) -> Dict:
        """
        Evaluate consistency across multiple responses to similar prompts.

        Args:
            responses: List of model responses to evaluate

        Returns:
            Dictionary with consistency metrics
        """
        if len(responses) < 2:
            return {'consistency_score': 1.0, 'variance': 0.0}

        # Compute pairwise semantic similarities
        embeddings = self.similarity_model.encode(responses, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)

        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices(len(responses), 1)]

        return {
            'consistency_score': np.mean(upper_triangle).item(),
            'variance': np.var(upper_triangle).item(),
            'min_similarity': np.min(upper_triangle).item(),
            'max_similarity': np.max(upper_triangle).item()
        }

    def logical_consistency(self, responses: List[str]) -> Dict:
        """
        Evaluate logical consistency across responses using NLI.

        Args:
            responses: List of model responses to evaluate

        Returns:
            Dictionary with logical consistency metrics
        """
        if len(responses) < 2:
            return {'logical_consistency': 1.0}

        # Compare each pair of responses
        entailment_scores = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                result = self.entailment_model(
                    f"{responses[i]} [SEP] {responses[j]}",
                    return_all_scores=True
                )
                # Score for entailment (label 0)
                entailment_scores.append(result[0][0]['score'])

        return {
            'logical_consistency': np.mean(entailment_scores).item(),
            'logical_variance': np.var(entailment_scores).item()
        }

    def self_consistency(self, response: str) -> Dict:
        """
        Evaluate self-consistency within a single response.

        Args:
            response: Model response to evaluate

        Returns:
            Dictionary with self-consistency metrics
        """
        # Split response into claims
        claims = self._extract_claims(response)
        if len(claims) < 2:
            return {'self_consistency': 1.0}

        # Check pairwise consistency
        consistent_pairs = 0
        total_pairs = 0

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                result = self.entailment_model(
                    f"{claims[i]} [SEP] {claims[j]}",
                    return_all_scores=True
                )
                # Considered consistent if not contradictory
                if result[0][2]['score'] < 0.5:  # Contradiction score
                    consistent_pairs += 1
                total_pairs += 1

        return {
            'self_consistency': consistent_pairs / total_pairs if total_pairs > 0 else 1.0,
            'num_claims': len(claims),
            'checked_pairs': total_pairs
        }

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract discrete claims from text.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted claims
        """
        # Simple sentence splitting - could be enhanced with NLP
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[:10]  # Limit to first 10 sentences for efficiency