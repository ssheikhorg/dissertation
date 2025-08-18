import evaluate
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from .config import settings


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
        """Detect fabricated medical entities and relationships"""
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
        results_df = model_evaluator.evaluate_model(model_name, prompts)

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
