# config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, Optional


class ModelConfig(BaseSettings):
    api_key: str
    model_name: str = Field(default="gpt-4")
    enable_rag: bool = Field(default=True)
    temperature: float = Field(default=0.3, ge=0, le=1)
    max_tokens: int = Field(default=1000, gt=0)
    use_gpu: bool = Field(default=True)


class EvaluationConfig(BaseSettings):
    use_gpu: bool = Field(default=True)
    similarity_model: str = Field(default="all-mpnet-base-v2")
    entailment_model: str = Field(default="roberta-large-mnli")
    medical_nli_model: str = Field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    ner_model: str = Field(default="d4data/biomedical-ner-all")
    umls_api_key: Optional[str] = None


class DatasetConfig(BaseSettings):
    batch_size: int = Field(default=5)
    rate_limits: Dict[str, float] = Field(
        default={"openai": 0.5, "anthropic": 1.0, "google": 2.0}
    )
    max_samples: int = Field(default=100)


class Settings(BaseSettings):
    models: Dict[str, ModelConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    pubmed_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"  # Ignore extra env variables


settings = Settings()
