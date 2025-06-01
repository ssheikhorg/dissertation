from models.api_models import OpenAIClient, AnthropicClient
from data.dataset_loader import DatasetLoader
from evaluation.evaluator import ModelEvaluator
from visualization.radar_plot import create_radar_plot
import yaml


def main():
    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize components
    dataset_loader = DatasetLoader()
    evaluator = ModelEvaluator()

    # Load test data
    test_data = dataset_loader.get_test_prompts("truthful_qa", n_samples=50)

    # Initialize models
    models = {
        "GPT-4": OpenAIClient(config["models"]["openai"]),
        "Claude": AnthropicClient(config["models"]["anthropic"]),
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        df = evaluator.evaluate_model(name, test_data, model)
        results[name] = {
            "accuracy": df["exact_match"].mean(),
            "consistency": 1 - df["std_dev"].mean(),
            "toxicity": df["toxicity_score"].mean(),
        }

    # Visualize results
    create_radar_plot(results, ["accuracy", "consistency", "toxicity"])


if __name__ == "__main__":
    main()
