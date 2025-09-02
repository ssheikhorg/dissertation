# evaluators.py - Hallucination evaluation with LangChain
import numpy as np
import re
from typing import List, Dict, Any
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import Criteria


class HallucinationEvaluator:
    """Evaluator focused on hallucination detection using LangChain"""

    def __init__(self):
        # Initialize LangChain evaluators
        self.factuality_evaluator = load_evaluator(
            "labeled_criteria",
            criteria=Criteria.FACTUALITY
        )

        # Medical facts for verification
        self.medical_knowledge_base = {
            "diabetes": ["increased thirst", "frequent urination", "fatigue", "blurred vision"],
            "aspirin": ["pain relief", "anti-inflammatory", "blood thinner", "fever reducer"],
            "hypertension": ["high blood pressure", "silent killer", "cardiovascular risk"],
            "cancer": ["abnormal cell growth", "tumors", "metastasis", "treatment options"],
            "covid": ["fever", "cough", "shortness of breath", "loss of taste/smell"]
        }

    async def evaluate_model(self, client, prompts: List[Dict], dataset: str, sample_count: int) -> Dict[str, Any]:
        """Evaluate a model for hallucinations using LangChain"""
        results = {
            "model": client.model_name,
            "dataset": dataset,
            "sample_count": sample_count,
            "metrics": {},
            "sample_responses": []
        }

        hallucination_scores = []
        accuracy_scores = []
        confidence_scores = []

        for prompt in prompts:
            response = client.generate(prompt["clean_prompt"])

            # Calculate metrics using LangChain evaluators
            accuracy = self._calculate_accuracy(prompt["clean_reference"], response)
            hallucination_score = await self._evaluate_factuality(prompt["clean_reference"], response)
            confidence = self._calculate_confidence(response)

            accuracy_scores.append(accuracy)
            hallucination_scores.append(hallucination_score)
            confidence_scores.append(confidence)

            results["sample_responses"].append({
                "prompt": prompt["original_prompt"],
                "reference": prompt["original_reference"],
                "response": response,
                "accuracy": accuracy,
                "hallucination_score": hallucination_score,
                "confidence": confidence
            })

        # Calculate aggregate metrics
        if accuracy_scores and hallucination_scores:
            results["metrics"] = {
                "accuracy": np.mean(accuracy_scores),
                "hallucination_rate": np.mean(hallucination_scores),
                "confidence": np.mean(confidence_scores),
                "response_length": np.mean([len(r["response"]) for r in results["sample_responses"]]),
                "consistency": 1.0 - np.std(hallucination_scores)  # Lower std = more consistent
            }

        return results

    async def _evaluate_factuality(self, reference: str, response: str) -> float:
        """Use LangChain to evaluate factuality"""
        try:
            # Use LangChain's factuality evaluator
            eval_result = self.factuality_evaluator.evaluate_strings(
                prediction=response,
                reference=reference,
                input="Medical question"
            )

            # Convert score to 0-1 scale (1 = completely factual)
            return 1.0 - (eval_result.get('score', 0.5) / 5.0)  # Assuming 5-point scale

        except Exception:
            # Fallback to pattern-based detection
            return self._pattern_based_hallucination_detection(response)

    def _pattern_based_hallucination_detection(self, response: str) -> float:
        """Fallback hallucination detection using patterns"""
        score = 0.0
        response_lower = response.lower()

        # Uncertainty indicators
        uncertainty_patterns = [
            r"\b(I think|I believe|probably|maybe|perhaps|likely)\b",
            r"\b(studies show|research indicates|experts say)\b",
        ]

        for pattern in uncertainty_patterns:
            matches = re.findall(pattern, response_lower)
            score += len(matches) * 0.1

        # Overgeneralizations
        overgeneralizations = re.findall(r"\b(always|never|every|all|none)\b", response_lower)
        score += len(overgeneralizations) * 0.15

        # Sensational claims
        sensational_claims = re.findall(r"\b(cure|miracle|breakthrough|revolutionary)\b", response_lower)
        score += len(sensational_claims) * 0.2

        return min(score, 1.0)

    def _calculate_accuracy(self, reference: str, response: str) -> float:
        """Calculate accuracy based on factual overlap"""
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())

        if not ref_words:
            return 0.0

        common_words = ref_words.intersection(resp_words)
        return len(common_words) / len(ref_words)

    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics"""
        confidence = 1.0
        response_lower = response.lower()

        # Reduce confidence for uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "I think", "I believe", "probably"]
        for marker in uncertainty_markers:
            if marker in response_lower:
                confidence -= 0.1

        # Reduce confidence for short or vague responses
        if len(response.split()) < 5:
            confidence -= 0.2

        return max(confidence, 0.1)  # Minimum confidence of 0.1


def generate_improvement_suggestions(metrics: Dict[str, float]) -> List[Dict[str, str]]:
    """Generate suggestions based on evaluation results"""
    suggestions = []

    hallucination_rate = metrics.get("hallucination_rate", 0)
    accuracy = metrics.get("accuracy", 0)
    confidence = metrics.get("confidence", 0)

    if hallucination_rate > 0.3:
        suggestions.append({
            "category": "High Priority",
            "suggestion": "Implement RAG with verified medical knowledge base",
            "expected_impact": "40-60% reduction in hallucinations"
        })

    if accuracy < 0.6:
        suggestions.append({
            "category": "High Priority",
            "suggestion": "Fine-tune with curated medical QA pairs and implement fact-checking",
            "expected_impact": "30-50% accuracy improvement"
        })

    if confidence < 0.6:
        suggestions.append({
            "category": "Medium Priority",
            "suggestion": "Add confidence calibration and uncertainty quantification",
            "expected_impact": "Better reliability estimation and fewer overconfident errors"
        })

    if metrics.get("consistency", 0) < 0.7:
        suggestions.append({
            "category": "Medium Priority",
            "suggestion": "Implement response consistency checks and self-verification",
            "expected_impact": "More consistent and reliable responses"
        })

    # Always include general best practices
    suggestions.append({
        "category": "General",
        "suggestion": "Implement multi-step verification: claim extraction → fact checking → response generation",
        "expected_impact": "Overall quality and reliability improvement"
    })

    return suggestions