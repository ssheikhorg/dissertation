
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import spacy
from config import settings
from data import DatasetLoader, get_dataset_stats
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from scipy.stats import stats
import torch
from sklearn.cluster import DBSCAN
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        """Encode texts into embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling to get sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)

        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy()

    def cosine_similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity between embeddings"""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)

        # Ensure both are tensors
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)

        # Normalize embeddings
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        return similarity


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

    def _calculate_exact_match(self, references: list[str], predictions: list[str]) -> float:
        """Calculate exact match score"""
        matches = 0
        for ref, pred in zip(references, predictions):
            if ref.strip().lower() == pred.strip().lower():
                matches += 1
        return matches / len(references) if references else 0.0


class ConsistencyEvaluator:
    def __init__(self, config: dict):
        """Initialize consistency evaluation tools.

        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.similarity_model = EmbeddingModel("sentence-transformers/all-mpnet-base-v2")

        self.entailment_model = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=0 if config.get("use_gpu", True) else -1,
        )

    def response_consistency(self, responses: list[str]) -> dict:
        if len(responses) < 2:
            return {"consistency_score": 1.0, "variance": 0.0}

        # Compute pairwise semantic similarities
        embeddings = self.similarity_model.encode(responses, convert_to_tensor=True)
        similarity_matrix = self.similarity_model.cosine_similarity(embeddings, embeddings).cpu().numpy()

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
        self.similarity_model = EmbeddingModel("sentence-transformers/all-mpnet-base-v2")

        self.entailment_id = 0  # MNLI entailment label index
        self.contradiction_id = 2  # MNLI contradiction label index

        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleurt = evaluate.load("bleurt", "bleurt-large-512")

        dataset_loader = DatasetLoader()
        dataset_loader.load_medical_entities()
        self.diseases = dataset_loader.diseases
        self.drugs = dataset_loader.drugs
        self.anatomy = dataset_loader.anatomy

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
        cos_sim = self.similarity_model.cosine_similarity(
            source_embedding.unsqueeze(0),
            gen_embedding.unsqueeze(0)
        ).item()

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
        """Detect fabricated medical entities and relationships"""
        entities = self.ner(generated)
        fabricated = []
        for entity in entities:
            if entity["entity_group"] in ["DISEASE", "DRUG", "ANATOMY"]:
                term = entity["word"]
                if (
                    (entity["entity_group"] == "DISEASE" and term not in self.diseases)
                    or (entity["entity_group"] == "DRUG" and term not in self.drugs)
                    or (entity["entity_group"] == "ANATOMY" and term not in self.anatomy)
                ):
                    fabricated.append(
                        {
                            "entity": term,
                            "type": entity["entity_group"],
                            "confidence": entity["score"],
                        }
                    )
        return {
            "fabricated_entities": fabricated,
            "fabrication_score": len(fabricated) / (len(entities) + 1e-6),
        }

    def check_clinical_guidelines(self, generated: str) -> dict:
        """Verify generated content against clinical guidelines"""
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


class ModelEvaluator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.metric_calc = MetricCalculator()
        self.bias_analyzer = BiasAnalyzer()
        self.consistency_eval = ConsistencyEvaluator(self.config)

    def evaluate_model(self, model_name: str, prompts: list[dict], client: "ModelClient") -> pd.DataFrame:
        model_client = client.get_client(model_name)
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


class MedicalModelEvaluator:
    def __init__(self):
        """Initialize medical-specific evaluator with healthcare-focused metrics"""
        self.hallucination_eval = HallucinationEvaluator()
        self.metric_calc = MetricCalculator()
        self.consistency_eval = ConsistencyEvaluator(settings.evaluation)

    def evaluate(self, model_client, prompts: list[dict]) -> dict:
        """Evaluate model performance on medical prompts

        Args:
            model_client: Initialized model client
            prompts: List of preprocessed prompts

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        results = {
            "total_prompts": len(prompts),
            "hallucination_rate": 0,
            "accuracy": 0,
            "fact_score": 0,
            "medical_bleu": 0,
            "toxicity_score": 0,
            "fabrication_score": 0,
            "guideline_violations": 0,
        }

        # Aggregate metrics across all prompts
        all_metrics = []

        for prompt in prompts:
            response = model_client.generate(prompt["clean_prompt"])

            # Basic text metrics
            metrics = self.metric_calc.calculate_all([prompt["clean_reference"]], [response])

            # Medical-specific evaluations
            hallucination = self.hallucination_eval.factual_consistency(prompt["clean_reference"], response)

            fabrication = self.hallucination_eval.detect_medical_fabrications(response)
            guidelines = self.hallucination_eval.check_clinical_guidelines(response)

            # Aggregate results
            all_metrics.append(
                {
                    **metrics,
                    **hallucination,
                    **fabrication,
                    **guidelines,
                    "response": response,
                }
            )

        # Calculate aggregate scores
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            results.update(
                {
                    "hallucination_rate": 1 - df["fact_score"].mean(),
                    "accuracy": df["exact_match"].mean(),
                    "fact_score": df["fact_score"].mean(),
                    "medical_bleu": df.get("medical_bleu", 0).mean(),
                    "toxicity_score": df.get("toxicity_score", 0).mean(),
                    "fabrication_score": df.get("fabrication_score", 0).mean(),
                    "guideline_violations": df.get("guideline_violations", 0).mean(),
                    "sample_responses": df["response"].iloc[:3].tolist(),  # Include sample responses
                }
            )

        return results


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


def compare_models(model_names: list[str], prompts: list[dict]) -> dict:
    """Compare multiple models on the same set of prompts

    Args:
        model_names: List of model names to compare
        prompts: List of preprocessed prompts to evaluate on

    Returns:
        Dictionary containing comparison results with:
        - Individual model results
        - Comparative analysis
        - Statistical significance
    """
    # Validate input
    if not model_names:
        raise ValueError("At least one model name must be provided")

    if not prompts:
        raise ValueError("No prompts provided for evaluation")

    # Evaluate each model (parallelized for efficiency)
    with ThreadPoolExecutor() as executor:
        futures = {
            model_name: executor.submit(evaluate_single_model, model_name, prompts) for model_name in model_names
        }
        all_results = {model_name: future.result() for model_name, future in futures.items()}

    # Generate comparative analysis
    comparison = generate_comparison_metrics(all_results)

    return {
        "models": all_results,
        "comparison": comparison,
        "prompt_count": len(prompts),
        "dataset_stats": get_dataset_stats(prompts),
    }


def evaluate_single_model(model_name: str, prompts: list[dict]) -> dict:
    """Evaluate a single model on the given prompts"""
    try:
        # Get per-prompt results
        results_df = ModelEvaluator().evaluate_model(model_name, prompts)

        # Calculate aggregate metrics
        return {
            "metrics": calculate_aggregate_metrics(results_df),
            "sample_responses": get_sample_responses(results_df),
            "error_analysis": analyze_errors(results_df),
        }
    except Exception as e:
        return {"error": str(e), "metrics": None, "sample_responses": None}


def calculate_aggregate_metrics(results_df: pd.DataFrame) -> dict:
    """Calculate aggregate metrics from individual results"""
    if results_df.empty:
        return {}

    return {
        "accuracy": results_df["exact_match"].mean(),
        "hallucination_rate": 1 - results_df["fact_score"].mean(),
        "toxicity_score": results_df["toxicity_score"].mean(),
        "consistency": results_df["self_consistency"].mean(),
        "rougeL": results_df["rougeL"].mean(),
        "medical_bleu": results_df.get("medical_bleu", np.nan).mean(),
        "response_length": results_df["response"].str.len().mean(),
    }


def get_sample_responses(results_df: pd.DataFrame, n: int = 3) -> list[dict]:
    """Get sample responses with reference comparisons"""
    samples = []
    for _, row in results_df.sample(min(n, len(results_df))).iterrows():
        samples.append(
            {
                "prompt": row["prompt"],
                "reference": row["reference"],
                "response": row["response"],
                "fact_score": row["fact_score"],
                "is_contradiction": row.get("is_contradiction", False),
            }
        )
    return samples


def analyze_errors(results_df: pd.DataFrame) -> dict:
    """Analyze common error patterns"""
    if results_df.empty:
        return {}

    # Get examples where fact_score < 0.5 (likely hallucinations)
    hallucinations = results_df[results_df["fact_score"] < 0.5]

    return {
        "hallucination_examples": get_sample_responses(hallucinations, 2) if not hallucinations.empty else [],
        "common_error_patterns": find_common_patterns(hallucinations["response"]),
        "hallucination_rate": len(hallucinations) / len(results_df),
    }


def find_common_patterns(responses: pd.Series) -> list[str]:
    """Identify common patterns in erroneous responses"""
    # This is a placeholder - implement proper NLP analysis
    from collections import Counter

    from nltk import ngrams

    # Simple n-gram analysis
    all_ngrams = []
    for text in responses:
        tokens = text.lower().split()
        all_ngrams.extend(ngrams(tokens, 3))

    return [" ".join(gram) for gram, count in Counter(all_ngrams).most_common(3)]


def generate_comparison_metrics(all_results: dict[str, dict]) -> dict:
    """Generate comparative metrics between models"""
    comparison = {}
    metrics = ["accuracy", "hallucination_rate", "consistency", "rougeL"]

    for metric in metrics:
        values = {
            model: results["metrics"][metric] for model, results in all_results.items() if results.get("metrics")
        }

        if values:
            comparison[metric] = {
                "best_model": max(values.items(), key=lambda x: x[1])[0],
                "worst_model": min(values.items(), key=lambda x: x[1])[0],
                "range": max(values.values()) - min(values.values()),
                "mean": np.mean(list(values.values())),
                "values": values,
            }

    return comparison


def generate_improvement_suggestions(current_results: dict, baseline: dict) -> list:
    """Generate specific suggestions to reduce hallucinations"""
    suggestions = []

    # Hallucination reduction suggestions
    if current_results["hallucination_rate"] > 0.2:
        suggestions.append(
            {
                "category": "High Priority",
                "suggestion": "Implement RAG (Retrieval Augmented Generation) to ground responses in verified medical literature",
                "expected_impact": "30-50% reduction in hallucinations",
            }
        )

    if current_results.get("fabrication_score", 0) > 0.1:
        suggestions.append(
            {
                "category": "Medium Priority",
                "suggestion": "Add entity verification against medical knowledge bases",
                "expected_impact": "20-40% reduction in fabricated entities",
            }
        )

    if current_results.get("guideline_violations", 0) > 0.15:
        suggestions.append(
            {
                "category": "High Priority",
                "suggestion": "Integrate clinical guideline checking using NLI models",
                "expected_impact": "40-60% reduction in guideline violations",
            }
        )

    # General suggestions based on performance
    accuracy_gap = baseline["accuracy"] - current_results["accuracy"]
    if accuracy_gap > 0.1:
        suggestions.append(
            {
                "category": "Medium Priority",
                "suggestion": "Fine-tune with medical QA pairs and implement consistency checks",
                "expected_impact": f"{round(accuracy_gap * 100)}% accuracy improvement",
            }
        )

    # Add model-specific suggestions
    suggestions.append(
        {
            "category": "General",
            "suggestion": "Implement multi-step verification: claim extraction → fact checking → response generation",
            "expected_impact": "25-45% overall improvement",
        }
    )

    return suggestions


def prepare_bar_chart_data(general_df: pd.DataFrame, metric: str, models: list[str] = None) -> dict[str, Any]:
    """Prepare data for bar chart visualization"""
    if general_df.empty:
        return {"error": "No data available for visualization"}

    # Filter by specific models if provided
    filtered_df = general_df
    if models and len(models) > 0:
        filtered_df = general_df[general_df["model"].isin(models)]

    # Group by model and calculate average for the selected metric
    model_avgs = filtered_df.groupby("model")[metric].mean().reset_index()

    # Prepare chart data
    chart_data = {
        "type": "bar",
        "data": {
            "labels": model_avgs["model"].tolist(),
            "datasets": [
                {
                    "label": metric.replace("_", " ").title(),
                    "data": model_avgs[metric].tolist(),
                    "backgroundColor": [],
                    "borderColor": [],
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{metric.replace('_', ' ').title()} Comparison",
                    "font": {"size": 16},
                }
            },
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": metric.replace("_", " ").title()}}
            },
        },
    }

    # Assign colors based on metric value
    for value in model_avgs[metric]:
        if "hallucination" in metric or "toxicity" in metric:
            # Red scale for negative metrics
            intensity = min(1.0, value * 3)
            chart_data["data"]["datasets"][0]["backgroundColor"].append(f"rgba(231, 76, 60, {0.6 + intensity * 0.4})")
            chart_data["data"]["datasets"][0]["borderColor"].append("rgba(231, 76, 60, 1)")
        else:
            # Green scale for positive metrics
            intensity = min(1.0, value * 1.5)
            chart_data["data"]["datasets"][0]["backgroundColor"].append(f"rgba(46, 204, 113, {0.6 + intensity * 0.4})")
            chart_data["data"]["datasets"][0]["borderColor"].append("rgba(46, 204, 113, 1)")

    return chart_data


def prepare_radar_data(general_df: pd.DataFrame, models: list[str], metrics: list[str]) -> dict[str, Any]:
    """Prepare data for radar chart visualization"""
    if general_df.empty:
        return {"error": "No data available for visualization"}

    # Default metrics if not specified
    if not metrics or len(metrics) == 0:
        metrics = ["hallucination_rate", "accuracy", "fact_score", "toxicity_score"]

    # Filter by specific models if provided
    selected_models = models if models and len(models) > 0 else general_df["model"].unique().tolist()

    chart_data = {
        "type": "radar",
        "data": {"labels": [metric.replace("_", " ").title() for metric in metrics], "datasets": []},
        "options": {
            "responsive": True,
            "plugins": {"title": {"display": True, "text": "Model Performance Radar Chart", "font": {"size": 16}}},
            "scales": {"r": {"beginAtZero": True, "max": 1, "ticks": {"stepSize": 0.2}}},
        },
    }

    # Colors for different models
    colors = [
        "rgba(255, 99, 132, 0.6)",  # Red
        "rgba(54, 162, 235, 0.6)",  # Blue
        "rgba(255, 206, 86, 0.6)",  # Yellow
        "rgba(75, 192, 192, 0.6)",  # Green
        "rgba(153, 102, 255, 0.6)",  # Purple
        "rgba(255, 159, 64, 0.6)",  # Orange
    ]

    # Calculate averages for each model and metric
    for i, model in enumerate(selected_models):
        model_data = general_df[general_df["model"] == model]
        averages = []

        for metric in metrics:
            values = model_data[metric].dropna()
            avg = values.mean() if len(values) > 0 else 0

            # For negative metrics, invert for radar (so center is good)
            if "hallucination" in metric or "toxicity" in metric:
                averages.append(1 - avg)  # Invert so lower is better
            else:
                averages.append(avg)

        color_idx = i % len(colors)
        chart_data["data"]["datasets"].append(
            {
                "label": model,
                "data": averages,
                "backgroundColor": colors[color_idx],
                "borderColor": colors[color_idx].replace("0.6", "1"),
                "borderWidth": 2,
                "pointBackgroundColor": colors[color_idx].replace("0.6", "1"),
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": colors[color_idx].replace("0.6", "1"),
            }
        )

    return chart_data


def prepare_scatter_data(
    general_df: pd.DataFrame, primary_metric: str, secondary_metric: str = None
) -> dict[str, Any]:
    """Prepare data for scatter plot visualization"""
    if general_df.empty:
        return {"error": "No data available for visualization"}

    # Default secondary metric if not specified
    if not secondary_metric:
        secondary_metric = "accuracy" if "hallucination" in primary_metric else "hallucination_rate"

    chart_data = {
        "type": "scatter",
        "data": {"datasets": []},
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{primary_metric.replace('_', ' ').title()} vs {secondary_metric.replace('_', ' ').title()}",
                    "font": {"size": 16},
                },
                "tooltip": {
                    "callbacks": {
                        "label": "function(context) { return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`; }"
                    }
                },
            },
            "scales": {
                "x": {
                    "title": {"display": True, "text": primary_metric.replace("_", " ").title()},
                    "beginAtZero": True,
                    "max": 1,
                },
                "y": {
                    "title": {"display": True, "text": secondary_metric.replace("_", " ").title()},
                    "beginAtZero": True,
                    "max": 1,
                },
            },
        },
    }

    # Group data by model
    models = general_df["model"].unique()
    colors = [
        "rgba(255, 99, 132, 0.8)",  # Red
        "rgba(54, 162, 235, 0.8)",  # Blue
        "rgba(255, 206, 86, 0.8)",  # Yellow
        "rgba(75, 192, 192, 0.8)",  # Green
        "rgba(153, 102, 255, 0.8)",  # Purple
        "rgba(255, 159, 64, 0.8)",  # Orange
    ]

    for i, model in enumerate(models):
        model_data = general_df[general_df["model"] == model]
        points = []

        for _, row in model_data.iterrows():
            if pd.notna(row[primary_metric]) and pd.notna(row[secondary_metric]):
                points.append({"x": float(row[primary_metric]), "y": float(row[secondary_metric])})

        if points:
            color_idx = i % len(colors)
            chart_data["data"]["datasets"].append(
                {
                    "label": model,
                    "data": points,
                    "backgroundColor": colors[color_idx],
                    "borderColor": colors[color_idx].replace("0.8", "1"),
                    "borderWidth": 1,
                    "pointRadius": 8,
                    "pointHoverRadius": 10,
                }
            )

    return chart_data
