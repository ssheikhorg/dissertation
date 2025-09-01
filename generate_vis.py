# create_demo_data.py
import asyncio
import json
from pathlib import Path

from app.routes import generate_visualization_data, prepare_bar_chart_data, prepare_radar_data, prepare_scatter_data


async def generate_demo_visualizations():
    """Pre-generate visualizations for the demo"""
    free_models = ["llama-2-7b", "mistral-7b", "qwen-7b", "meditron-7b", "biomedgpt"]
    metrics = ["accuracy", "hallucination_rate", "fact_score", "toxicity_score"]
    viz_types = ["bar", "radar", "scatter"]

    # Generate data once
    general_df, bias_df = await generate_visualization_data(
        model_names=free_models,
        metrics=metrics,
        n_samples=20,  # Smaller sample for demo
    )

    # Generate all combinations
    for viz_type in viz_types:
        for metric in metrics:
            if viz_type == "bar":
                chart_data = prepare_bar_chart_data(general_df, metric, free_models)
            elif viz_type == "radar":
                chart_data = prepare_radar_data(general_df, free_models, [metric])
            elif viz_type == "scatter":
                secondary_metric = "accuracy" if "hallucination" in metric else "hallucination_rate"
                chart_data = prepare_scatter_data(general_df, metric, secondary_metric)

            # Save to cache
            cache_file = Path("cache") / "visualizations" / f"demo_{viz_type}_{metric}.json"
            with open(cache_file, "w") as f:
                json.dump(chart_data, f, indent=2)

    print("Demo visualizations generated successfully!")


if __name__ == "__main__":
    asyncio.run(generate_demo_visualizations())
