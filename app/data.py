from typing import List, Dict, Any


class ModelsEnum:
    """Enum for supported models"""
    LLAMA_2_7B = "llama-2-7b"
    MISTRAL_7B = "mistral-7b"
    QWEN_7B = "qwen-7b"
    MEDITRON_7B = "meditron-7b"
    BIOMEDGPT = "biomedgpt"
    GPT_4 = "gpt-4"
    CLAUDE_3_OPUS = "claude-3-opus"
    GROK = "grok"


class DatasetLoader:
    """Loader for medical evaluation datasets"""

    def __init__(self):
        self.fallback_prompts = [
            {
                "question": "What are common diabetes symptoms?",
                "long_answer": "Common diabetes symptoms include increased thirst, frequent urination, and fatigue.",
            },
            {
                "question": "How does aspirin work?",
                "long_answer": "Aspirin reduces pain and inflammation by blocking certain enzymes.",
            }
        ]

    def load_test_prompts(self, dataset_name: str, n_samples: int = 5) -> List[Dict[str, str]]:
        """Load test prompts - simplified version"""
        # In a real implementation, this would load from actual datasets
        # For now, using fallback prompts

        prompts = self.fallback_prompts[:n_samples]

        # Preprocess prompts
        processed = []
        for prompt in prompts:
            processed.append({
                "original_prompt": prompt["question"],
                "clean_prompt": self.clean_text(prompt["question"]),
                "original_reference": prompt["long_answer"],
                "clean_reference": self.clean_text(prompt["long_answer"])
            })

        return processed

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        return text.lower().strip()


# Global instance
dataset_loader = DatasetLoader()


def load_test_prompts(dataset_name: str, n_samples: int = 5) -> List[Dict[str, str]]:
    """Convenience function to load test prompts"""
    return dataset_loader.load_test_prompts(dataset_name, n_samples)


def load_baseline(model_name: str, dataset_name: str) -> Dict[str, float]:
    """Load baseline metrics for a model"""
    # Simple baseline values
    return {
        "accuracy": 0.7,
        "hallucination_rate": 0.25,
        "fact_score": 0.75
    }