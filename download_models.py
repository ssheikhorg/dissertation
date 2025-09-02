from huggingface_hub import snapshot_download
from pathlib import Path

models = [
    '7B-Instruct-v0.3',
    'reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B'
]
path = r'D:\models'
models_path = Path(path).joinpath()
models_path.mkdir(parents=True, exist_ok=True)

try:
    snapshot_download(
        repo_id="reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B",
        local_dir=models_path
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Trying alternative download approach...")
