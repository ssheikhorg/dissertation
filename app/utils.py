from typing import Any

from fastapi import HTTPException


async def generate_comparison_metrics(models_data: dict[str, Any]) -> dict[str, Any]:
    """Generate comparative metrics between models"""
    comparison = {}
    metrics_to_compare = ["accuracy", "hallucination_rate", "confidence", "response_length", "consistency"]

    for metric in metrics_to_compare:
        values = {}

        for model_name, data in models_data.items():
            if "metrics" in data and metric in data["metrics"]:
                values[model_name] = data["metrics"][metric]

        if values:
            # For accuracy and confidence, higher is better
            # For hallucination_rate, lower is better
            if metric in ["accuracy", "confidence", "consistency"]:
                best_model = max(values.items(), key=lambda x: x[1])[0]
                worst_model = min(values.items(), key=lambda x: x[1])[0]
            else:  # hallucination_rate
                best_model = min(values.items(), key=lambda x: x[1])[0]
                worst_model = max(values.items(), key=lambda x: x[1])[0]

            comparison[metric] = {
                "best_model": best_model,
                "worst_model": worst_model,
                "range": max(values.values()) - min(values.values()),
                "average": sum(values.values()) / len(values),
                "values": values,
            }

    return comparison
