from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic import BaseModel, Field, validator, HttpUrl
from pydantic_settings import BaseSettings


# Sub-models for type-safe configuration
class ModelConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    default_model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=1)
    max_tokens: int = Field(1000, gt=0)
    rate_limit_delay: float = Field(1.0, gt=0)


class DatasetConfig(BaseModel):
    name: str
    split: str = "validation"
    sample_size: int = Field(100, gt=0)
    prompt_column: str = "question"
    reference_column: str = "best_answer"


class VisualizationConfig(BaseModel):
    theme: str = "plotly_dark"
    color_palette: str = "Set2"
    default_figsize: List[int] = [12, 6]
    save_format: str = "png"
    save_dpi: int = 300
    output_dir: Path = Path("./reports")


class Settings(BaseSettings):
    # Project metadata
    project_name: str = "CDN4-Applicability-Domains"
    author: str = "Shahin R Shapon"
    version: str = "1.0.0"

    # Model configurations
    openai: ModelConfig
    anthropic: ModelConfig
    google: ModelConfig
    huggingface: Dict[str, Any]  # More flexible for HF options

    # Evaluation parameters
    datasets: List[DatasetConfig]
    metrics: Dict[str, List[str]]
    thresholds: Dict[str, float]

    # Visualization settings
    visualization: VisualizationConfig

    # Runtime settings
    max_workers: int = 4
    cache_responses: bool = True
    log_level: str = "INFO"

    class Config:
        env_nested_delimiter = '__'
        extra = "ignore"

    @classmethod
    def from_yaml(cls, path: Path) -> 'Settings':
        """Load settings from YAML file"""
        with open(path) as f:
            config_data = yaml.safe_load(f)
        return cls(**cls._flatten_dict(config_data))

    @staticmethod
    def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict for Pydantic"""
        result = {}
        for key, value in data.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict):
                result.update(cls._flatten_dict(value, f"{full_key}_"))
            else:
                result[full_key] = value
        return result

    @validator("log_level")
    def validate_log_level(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("Invalid log level")
        return v.upper()


# Singleton instance
settings: Settings = None


def init_settings(config_path: Path = Path("config.yaml")) -> Settings:
    """Initialize and return the global settings"""
    global settings
    if settings is None:
        settings = Settings.from_yaml(config_path)
    return settings


def get_settings() -> Settings:
    """Get the initialized settings"""
    if settings is None:
        raise RuntimeError("Settings not initialized. Call init_settings() first.")
    return settings


# Initialize once at application start
init_settings()

# Access anywhere in your code
settings = get_settings()
