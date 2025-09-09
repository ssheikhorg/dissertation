import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")
EVALUATION_PATH = Path(__file__).parent / "model_evaluations"


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


@router.get("/api/evaluation/{model_name}")
async def get_evaluation_results(model_name: str):
    """Get evaluation results for a specific model from downloaded files"""
    try:
        base_dir = EVALUATION_PATH / model_name / "data"

        possible_files = [
            base_dir / f"{model_name}_comprehensive_results.json",
            base_dir / f"{model_name}_ui_export_data.json",
            *base_dir.glob("*.json"),  # Any other JSON files
        ]

        # Filter to only existing files
        existing_files = [f for f in possible_files if f.exists()]

        if not existing_files:
            raise HTTPException(status_code=404, detail=f"No evaluation results found for {model_name}")

        # Use the most recent file
        latest_file = max(existing_files, key=os.path.getctime)

        with open(latest_file) as f:
            results = json.load(f)

        # Handle different result formats
        results["dataset_metrics"] = results.get("dataset_metrics", results.get("dataset_details", {}))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")


@router.get("/api/dataset-metrics/{model_name}")
async def get_dataset_metrics(model_name: str, dataset: str = None):
    """Get dataset-specific metrics for a model"""
    try:
        results = await get_evaluation_results(model_name)

        # Extract dataset metrics
        dataset_metrics = results.get("dataset_metrics", {})

        if dataset:
            # Return specific dataset metrics if requested
            if dataset in dataset_metrics:
                return {"model": model_name, "dataset": dataset, "metrics": dataset_metrics[dataset]}
            else:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found for model {model_name}")
        else:
            return {"model": model_name, "dataset_metrics": dataset_metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/available")
async def get_available_models():
    """Get list of models that have evaluation results"""
    models = set()

    if not EVALUATION_PATH.exists():
        return {"models": []}

    for item in EVALUATION_PATH.iterdir():
        if item.is_dir():
            data_dir = item / "data"
            if data_dir.exists():
                json_files = list(data_dir.glob("*.json"))
                if json_files:
                    models.add(item.name)

    return {"models": sorted(list(models))}


@router.get("/api/visualization/{model_name}/{viz_type}")
async def get_visualization(model_name: str, viz_type: str):
    """Get pre-generated visualization images"""
    viz_mapping = {
        "accuracy_bar": "accuracy_bar_chart.png",
        "hallucination_rate_bar": "hallucination_rate_bar_chart.png",
        "confidence_bar": "confidence_bar_chart.png",
        "radar": "radar_chart.png",
        "comparison_table": "comparison_table.html",
        "dataset_comparison": "dataset_comparison_chart.png",
        "medqa_radar": "medqa_radar_chart.png",
        "mimic_cxr_radar": "mimic_cxr_radar_chart.png",
        "pubmedqa_radar": "pubmedqa_radar_chart.png",
    }
    if viz_type not in viz_mapping:
        raise HTTPException(status_code=404, detail=f"Visualization type {viz_type} not supported")

    file_name = viz_mapping[viz_type]
    file_path = EVALUATION_PATH / model_name / "charts" / file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization file {file_name} not found for model {model_name}")

    if viz_type == "comparison_table":
        with open(file_path) as f:
            content = f.read()
        return HTMLResponse(content)
    else:
        return FileResponse(file_path)


@router.get("/api/model-metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get just the metrics for a model"""
    try:
        results = await get_evaluation_results(model_name)
        # Extract metrics from different possible result structures
        if "evaluation_results" in results and "metrics" in results["evaluation_results"]:
            metrics = results["evaluation_results"]["metrics"]
        elif "metrics" in results:
            metrics = results["metrics"]
        else:
            metrics = {}

        # Include dataset metrics if available
        dataset_metrics = results.get("dataset_metrics", {})

        return {"model": model_name, "metrics": metrics, "dataset_metrics": dataset_metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/compare")
async def compare_models(model1: str, model2: str):
    """Compare two models using pre-generated results"""
    comparison_data = {}

    for model_name in [model1, model2]:
        try:
            metrics_data = await get_model_metrics(model_name)
            comparison_data[model_name] = metrics_data
        except Exception as e:
            comparison_data[model_name] = {"error": str(e)}

    # Generate comparative analysis
    comparison_metrics = await generate_comparison_metrics(comparison_data)

    return {"models": comparison_data, "comparison": comparison_metrics}
