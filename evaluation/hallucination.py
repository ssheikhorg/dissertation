from typing import Dict, List, Tuple
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datasets import load_metric
import spacy


class HallucinationEvaluator:
    def __init__(self, config: Dict):
        """
        Initialize hallucination evaluation tools.

        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")

        # Load models for different detection methods
        self.nli_model = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=0 if config.get("use_gpu", True) else -1
        )

        self.similarity_model = SentenceTransformer(
            'all-mpnet-base-v2',
            device='cuda' if config.get("use_gpu", True) else 'cpu'
        )

        self.entailment_id = 0  # MNLI entailment label index
        self.contradiction_id = 2  # MNLI contradiction label index

        # Load metrics
        self.rouge = load_metric('rouge')
        self.bleurt = load_metric('bleurt', 'bleurt-large-512')

    def detect_contradictions(self, source: str, generated: str) -> Dict:
        """
        Detect contradictions between source and generated text using NLI.

        Args:
            source: Original source text
            generated: Model-generated text

        Returns:
            Dictionary with contradiction scores
        """
        result = self.nli_model(
            f"{source} [SEP] {generated}",
            return_all_scores=True
        )

        entailment_score = result[0][self.entailment_id]['score']
        contradiction_score = result[0][self.contradiction_id]['score']

        return {
            'entailment_score': entailment_score,
            'contradiction_score': contradiction_score,
            'is_contradiction': contradiction_score > 0.5
        }

    def semantic_similarity(self, source: str, generated: str) -> Dict:
        """
        Compute semantic similarity between source and generated text.

        Args:
            source: Original source text
            generated: Model-generated text

        Returns:
            Dictionary with similarity metrics
        """
        # Encode sentences
        source_embedding = self.similarity_model.encode(source, convert_to_tensor=True)
        gen_embedding = self.similarity_model.encode(generated, convert_to_tensor=True)

        # Compute cosine similarity
        cos_sim = util.cos_sim(source_embedding, gen_embedding).item()

        # Compute ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=[generated],
            references=[source]
        )

        # Compute BLEURT score
        bleurt_score = self.bleurt.compute(
            predictions=[generated],
            references=[source]
        )['scores'][0]

        return {
            'cosine_similarity': cos_sim,
            'rougeL': rouge_scores['rougeL'].mid.fmeasure,
            'bleurt_score': bleurt_score
        }

    def factual_consistency(self, source: str, generated: str) -> Dict:
        """
        Comprehensive factual consistency evaluation.

        Args:
            source: Original source text
            generated: Model-generated text

        Returns:
            Dictionary with consistency metrics
        """
        # Extract claims from generated text
        generated_claims = self._extract_claims(generated)
        if not generated_claims:
            return {'fact_score': 1.0, 'num_claims': 0}

        # Verify each claim against source
        verified = []
        for claim in generated_claims:
            result = self.detect_contradictions(source, claim)
            verified.append(result['entailment_score'] > 0.5)

        fact_score = np.mean(verified)

        return {
            'fact_score': fact_score,
            'num_claims': len(generated_claims),
            'verified_claims_ratio': fact_score
        }

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using NLP techniques.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted factual claims
        """
        doc = self.nlp(text)
        claims = []

        # Extract simple declarative sentences
        for sent in doc.sents:
            if any(tok.dep_ == 'ROOT' for tok in sent):
                claims.append(sent.text)

        return claims