import asyncio

from app.clients import generate_visualization_data, create_grouped_barplot, metric_comparison_bars, stacked_bias_bars


# Example usage of generate_visualization_data
async def main():
    # Generate data for all registered models
    general_df, bias_df = await generate_visualization_data(
        model_names=None,  # Use all models
        datasets=["pubmed_qa", "med_qa"],
        metrics=["accuracy", "hallucination_rate", "toxicity_score"],
        bias_types=["Gender", "Racial", "Political"],
        n_samples=10  # Small for testing
    )

    # Plotting examples
    create_grouped_barplot(
        data=general_df,
        x_column="dataset",
        y_column="accuracy",
        hue_column="model",
        title="Accuracy Comparison Across Datasets",
    )

    metric_comparison_bars(
        results_df=general_df,
        metrics=["accuracy", "hallucination_rate"],
        title="Performance Metrics Comparison",
    )

    stacked_bias_bars(
        results_df=bias_df,
        bias_types=["Gender", "Racial", "Political"],
        title="Bias Composition Across Models",
    )

if __name__ == "__main__":
    asyncio.run(main())
