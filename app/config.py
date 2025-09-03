# config.py - Simplified configuration

class Settings():
    # Model settings
    model_name: str = "llama-2-7b"
    model_path: str = "D:/models"
    temperature: float = 0.3
    max_tokens: int = 1000

    # Device settings
    device: str = "auto"  # auto, cuda, cpu, mps
    use_fp16: bool = True

    # Evaluation settings
    batch_size: int = 5
    max_samples: int = 100

    # RAG settings
    enable_rag: bool = True
    rag_top_k: int = 3


settings = Settings()
