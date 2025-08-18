
import evaluate
import numpy as np
import pandas as pd
import scipy.stats as stats
import spacy
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from app.api_models import ModelClient
from app.config import settings


class BiasAnalyzer:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

    def analyze_responses(self, responses: list[str]) -> dict:
        results = self.classifier(responses)
        toxic_scores = [r["score"] if r["label"] == "toxic" else 0 for r in results]
        return {
            "toxicity_score": np.mean(toxic_scores),
            "max_toxicity": max(toxic_scores),
            "toxic_count": sum(score > 0.5 for score in toxic_scores),
        }


class ConsistencyEvaluator:
    def __init__(self, config: dict):
        """Initialize consistency evaluation tools.

        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.similarity_model = SentenceTransformer(
            "all-mpnet-base-v2", device="cuda" if config.get("use_gpu", True) else "cpu"
        )

        self.entailment_model = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=0 if config.get("use_gpu", True) else -1,
        )

    def response_consistency(self, responses: list[str]) -> dict:
        """Evaluate consistency across multiple responses to similar prompts.

        Args:
            responses: List of model responses to evaluate

        Returns:
            Dictionary with consistency metrics
        """
        if len(responses) < 2:
            return {"consistency_score": 1.0, "variance": 0.0}

        # Compute pairwise semantic similarities
        embeddings = self.similarity_model.encode(responses, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)

        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices(len(responses), 1)]

        return {
            "consistency_score": np.mean(upper_triangle).item(),
            "variance": np.var(upper_triangle).item(),
            "min_similarity": np.min(upper_triangle).item(),
            "max_similarity": np.max(upper_triangle).item(),
        }

    def logical_consistency(self, responses: list[str]) -> dict:
        """Evaluate logical consistency across responses using NLI.

        Args:
            responses: List of model responses to evaluate

        Returns:
            Dictionary with logical consistency metrics
        """
        if len(responses) < 2:
            return {"logical_consistency": 1.0}

        # Compare each pair of responses
        entailment_scores = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                result = self.entailment_model(f"{responses[i]} [SEP] {responses[j]}", return_all_scores=True)
                # Score for entailment (label 0)
                entailment_scores.append(result[0][0]["score"])

        return {
            "logical_consistency": np.mean(entailment_scores).item(),
            "logical_variance": np.var(entailment_scores).item(),
        }

    def self_consistency(self, response: str) -> dict:
        """Evaluate self-consistency within a single response.

        Args:
            response: Model response to evaluate

        Returns:
            Dictionary with self-consistency metrics
        """
        # Split response into claims
        claims = self._extract_claims(response)
        if len(claims) < 2:
            return {"self_consistency": 1.0}

        # Check pairwise consistency
        consistent_pairs = 0
        total_pairs = 0

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                result = self.entailment_model(f"{claims[i]} [SEP] {claims[j]}", return_all_scores=True)
                # Considered consistent if not contradictory
                if result[0][2]["score"] < 0.5:  # Contradiction score
                    consistent_pairs += 1
                total_pairs += 1

        return {
            "self_consistency": consistent_pairs / total_pairs if total_pairs > 0 else 1.0,
            "num_claims": len(claims),
            "checked_pairs": total_pairs,
        }

    def _extract_claims(self, text: str) -> list[str]:
        """Extract discrete claims from text.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted claims
        """
        # Simple sentence splitting - could be enhanced with NLP
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences[:10]  # Limit to first 10 sentences for efficiency


class ModelEvaluator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.metric_calc = MetricCalculator()
        self.bias_analyzer = BiasAnalyzer()
        self.consistency_eval = ConsistencyEvaluator(self.config)

    def evaluate_model(self, model_name: str, prompts: list[dict], mitigation: str = None) -> pd.DataFrame:
        model_client = ModelClient.get_client(model_name)
        results = []

        for prompt in prompts:
            response = model_client.generate(prompt["clean_prompt"])

            metrics = self.metric_calc.calculate_all([prompt["clean_reference"]], [response])

            bias = self.bias_analyzer.analyze_responses([response])
            self_consistency = self.consistency_eval.self_consistency(response)

            results.append(
                {
                    "model": model_name,
                    "prompt": prompt["original_prompt"],
                    "response": response,
                    **metrics,
                    **bias,
                    **self_consistency,
                }
            )

        return pd.DataFrame(results)

    def compare_models(self, model_names: list[str], prompts: list[dict]) -> pd.DataFrame:
        all_results = []
        for model_name in model_names:
            results = self.evaluate_model(model_name, prompts)
            all_results.append(results)
        return pd.concat(all_results)


class HallucinationEvaluator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nli_model = pipeline(
            "text-classification",
            model=settings.evaluation.entailment_model,
            device=0 if settings.evaluation.use_gpu else -1,
        )
        self.medical_nli = pipeline(
            "text-classification",
            model=settings.evaluation.medical_nli_model,
            device=0 if settings.evaluation.use_gpu else -1,
        )
        self.ner = pipeline(
            "ner",
            model=settings.evaluation.ner_model,
            device=0 if settings.evaluation.use_gpu else -1,
        )
        self.similarity_model = SentenceTransformer(
            settings.evaluation.similarity_model,
            device="cuda" if settings.evaluation.use_gpu else "cpu",
        )

        self.entailment_id = 0  # MNLI entailment label index
        self.contradiction_id = 2  # MNLI contradiction label index

        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleurt = evaluate.load("bleurt", "bleurt-large-512")

    def detect_contradictions(self, source: str, generated: str) -> dict:
        """Detect contradictions between source and generated text using NLI.

        Args:
            source: Original source text
            generated: Model-generated text

        Returns:
            Dictionary with contradiction scores
        """
        result = self.nli_model(f"{source} [SEP] {generated}", return_all_scores=True)

        entailment_score = result[0][self.entailment_id]["score"]
        contradiction_score = result[0][self.contradiction_id]["score"]

        return {
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "is_contradiction": contradiction_score > 0.5,
        }

    def semantic_similarity(self, source: str, generated: str) -> dict:
        """Compute semantic similarity between source and generated text.

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
        rouge_scores = self.rouge.compute(predictions=[generated], references=[source])

        # Compute BLEURT score
        bleurt_score = self.bleurt.compute(predictions=[generated], references=[source])["scores"][0]

        return {
            "cosine_similarity": cos_sim,
            "rougeL": rouge_scores["rougeL"].mid.fmeasure,
            "bleurt_score": bleurt_score,
        }

    def factual_consistency(self, source: str, generated: str) -> dict:
        """Comprehensive factual consistency evaluation.

        Args:
            source: Original source text
            generated: Model-generated text

        Returns:
            Dictionary with consistency metrics
        """
        # Extract claims from generated text
        generated_claims = self._extract_claims(generated)
        if not generated_claims:
            return {"fact_score": 1.0, "num_claims": 0}

        # Verify each claim against source
        verified = []
        for claim in generated_claims:
            result = self.detect_contradictions(source, claim)
            verified.append(result["entailment_score"] > 0.5)

        fact_score = np.mean(verified)

        return {
            "fact_score": fact_score,
            "num_claims": len(generated_claims),
            "verified_claims_ratio": fact_score,
        }

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text using NLP techniques.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted factual claims
        """
        doc = self.nlp(text)
        claims = []

        # Extract simple declarative sentences
        for sent in doc.sents:
            if any(tok.dep_ == "ROOT" for tok in sent):
                claims.append(sent.text)

        return claims

    def detect_medical_fabrications(self, generated: str) -> dict:
        """Detect fabricated medical entities and relationships
        """
        entities = self.ner(generated)
        fabricated = []

        for entity in entities:
            # Check if entity exists in medical knowledge bases
            if entity["entity_group"] in ["DISEASE", "DRUG", "ANATOMY"]:
                if not self.umls.exists(entity["word"]) and not self.mesh.exists(entity["word"]):
                    fabricated.append(
                        {
                            "entity": entity["word"],
                            "type": entity["entity_group"],
                            "confidence": entity["score"],
                        }
                    )

        return {
            "fabricated_entities": fabricated,
            "fabrication_score": len(fabricated) / (len(entities) + 1e-6),
        }

    def check_clinical_guidelines(self, generated: str) -> dict:
        """Verify generated content against clinical guidelines
        """
        # This would integrate with guideline databases like UpToDate or PubMed
        violations = []

        # Example implementation would use semantic search against guidelines
        guideline_matches = self.medical_nli(generated, return_all_scores=True)

        return {
            "guideline_violations": guideline_matches["contradictions"],
            "evidence_based_score": guideline_matches["entailments"],
        }

    def achmi_score(self, source_ents: list[str], gen_ents: list[str], captions: list[str]) -> dict:
        halluc_components = set(gen_ents) - set(source_ents)
        achmi_i = len(halluc_components) / len(gen_ents) if gen_ents else 0
        halluc_sentences = [any(e in halluc_components for e in caption.split()) for caption in captions]
        achmi_s = sum(halluc_sentences) / len(captions) if captions else 0
        return {"achmi_i": achmi_i, "achmi_s": achmi_s}

    def semantic_entropy(self, client, prompt: str, n_samples: int = 10) -> float:
        responses = [client.generate(prompt) for _ in range(n_samples)]
        embeddings = self.similarity_model.encode(responses)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
        labels = clustering.labels_
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        probs = counts / counts.sum() if counts.sum() > 0 else [1]
        return stats.entropy(probs)


class MetricCalculator:
    def __init__(self):
        """Initialize all metrics with error handling"""
        try:
            self.rouge = evaluate.load("rouge")
            self.bleu = evaluate.load("bleu")
            self.bertscore = evaluate.load("bertscore")
        except Exception as e:
            raise ImportError(
                "Metric loading failed. Ensure you've installed all required packages:\n"
                "pip install evaluate sacrebleu bert-score rouge-score nltk\n"
                f"Original error: {str(e)}"
            )

    def calculate_all(self, references: list[str], predictions: list[str]) -> dict:
        """Calculate all metrics between references and predictions"""
        # ROUGE (handles multiple references automatically)
        rouge_results = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)

        # BLEU (requires specific formatting)
        bleu_results = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],  # List of references per prediction
        )

        # BERTScore (computes similarity)
        bertscore_results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",  # Best for medical text
        )

        return {
            "rougeL": rouge_results["rougeL"].mid.fmeasure,  # Get F1 score
            "bleu": bleu_results["bleu"],
            "bertscore_f1": np.mean(bertscore_results["f1"]),
            "exact_match": self._calculate_exact_match(references, predictions),
        }


class MedicalLoRAFineTuner:
    def __init__(self, base_model: str, medical_dataset: str):
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.dataset = load_dataset(medical_dataset)

        # Initialize LoRA configuration
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def fine_tune(self, epochs=3):
        # Prepare dataset
        train_data = self.dataset.map(self._preprocess_function, batched=True)

        # Add LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)

        # Training setup
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=1e-4,
            weight_decay=0.01,
            fp16=True,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def _preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, max_length=512)
