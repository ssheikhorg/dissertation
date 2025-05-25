from models.api_models import OpenAIClient
from data.dataset_loader import load_benchmark
from evaluation.evaluator import run_evaluation


def main():
    # Initialize
    gpt = OpenAIClient(api_key=config.OPENAI_KEY)
    dataset = load_benchmark('truthful_qa')

    # Evaluate
    results = run_evaluation(
        models={'GPT-4': gpt},
        dataset=dataset
    )

    # Visualize
    plot_radar(results)


if __name__ == "__main__":
    main()
