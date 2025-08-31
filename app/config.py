from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
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
    def lora_config(self) -> dict[str, Any]:
        """Return LoRA configuration as a dictionary"""
        return {"r": self.lora_r, "alpha": self.lora_alpha, "dropout": self.lora_dropout}

    @property
    def rate_limits(self) -> dict[str, float]:
        """Return rate limits as a dictionary"""
        return {"openai": 0.5, "anthropic": 1.0, "google": 2.0, "huggingface": 1.0}

    @field_validator("use_gpu", "use_mps", mode="before")
    def validate_device_settings(cls, v, values):
        """Ensure only one device type is enabled"""
        if values.data.get("force_cpu", False):
            return False
        return v

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        env_prefix = "MODEL_"
        extra = "ignore"


settings = ModelConfig()
