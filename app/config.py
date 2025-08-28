from typing import Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

import torch
import os



class ModelConfig(BaseSettings):
    # API Keys (all optional with None default)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    xai_api_key: str | None = None
    pubmed_api_key: str | None = None
    umls_api_key: str | None = None

    # Model settings
    model_name: str = Field(default="gpt-4")
    enable_rag: bool = Field(default=True)
    temperature: float = Field(default=0.3, ge=0, le=1)
    max_tokens: int = Field(default=1000, gt=0)
    force_cpu: bool = Field(default=False)
    use_gpu: bool = Field(default=False)
    use_mps: bool = Field(default=True)  # Apple Silicon support

    # Evaluation models
    similarity_model: str = Field(default="all-mpnet-base-v2")
    entailment_model: str = Field(default="roberta-large-mnli")
    medical_nli_model: str = Field(default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ner_model: str = Field(default="d4data/biomedical-ner-all")

    # LoRA configuration
    lora_r: int = Field(default=8)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.05)

    # Evaluation settings
    entropy_samples: int = Field(default=10)

    # Dataset configuration
    batch_size: int = Field(default=5)
    max_samples: int = Field(default=100)

    # Threading configuration for Apple Silicon
    mps_threads: int = Field(default=4)
    cpu_threads: int = Field(default=8)
    use_fp16: bool = Field(default=True)

    @property
    def lora_config(self) -> Dict[str, Any]:
        """Return LoRA configuration as a dictionary"""
        return {"r": self.lora_r, "alpha": self.lora_alpha, "dropout": self.lora_dropout}

    @property
    def rate_limits(self) -> Dict[str, float]:
        """Return rate limits as a dictionary"""
        return {
            "openai": 0.5,
            "anthropic": 1.0,
            "google": 2.0,
            "huggingface": 1.0
        }

    @field_validator('use_gpu', 'use_mps', mode='before')
    def validate_device_settings(cls, v, values):
        """Ensure only one device type is enabled"""
        if values.data.get('force_cpu', False):
            return False
        return v

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        env_prefix = "MODEL_"
        extra = "ignore"


settings = ModelConfig()


def setup_device():
    """Configure PyTorch device settings for macOS compatibility"""
    # Set environment variables to prevent MPS issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    # Determine device
    if settings.force_cpu:
        device = "cpu"
        torch.set_num_threads(settings.cpu_threads)
    elif torch.cuda.is_available() and settings.use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and settings.use_mps:
        try:
            # Test MPS availability
            test_tensor = torch.ones(1, device="mps")
            device = "mps"
            # Set thread limits for MPS
            torch.set_num_threads(min(settings.mps_threads, 4))  # Limit threads for MPS
            print(f"MPS device initialized successfully")
        except Exception as e:
            print(f"MPS unavailable, falling back to CPU: {e}")
            device = "cpu"
            torch.set_num_threads(settings.cpu_threads)
    else:
        device = "cpu"
        torch.set_num_threads(settings.cpu_threads)

    print(f"Using device: {device}")
    return device


def cleanup_device():
    """Clean up device resources"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
