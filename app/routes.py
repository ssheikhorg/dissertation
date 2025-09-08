from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
import json
import os
import glob
from datetime import datetime

from .clients import ModelFactory
from .evaluators import HallucinationEvaluator, generate_improvement_suggestions
from .data import load_test_prompts, load_baseline

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Directory where evaluation results are stored
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory for pre-generated visualizations
VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


@router.get("/api/evaluation/{model_name}")
async def get_evaluation_results(model_name: str):
    """Get evaluation results for a specific model from downloaded files"""
    try:
        # Look for the comprehensive results file
        pattern = os.path.join(RESULTS_DIR, f"{model_name}_comprehensive_results.json")
        files = glob.glob(pattern)

        if not files:
            # Fallback to UI export data
            pattern = os.path.join(RESULTS_DIR, f"{model_name}_ui_export_data.json")
            files = glob.glob(pattern)

        if not files:
            raise HTTPException(status_code=404, detail=f"No evaluation results found for {model_name}")

        # Use the most recent file
        latest_file = max(files, key=os.path.getctime)

        with open(latest_file, "r") as f:
            results = json.load(f)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")


@router.get("/api/models/available")
async def get_available_models():
    """Get list of models that have evaluation results"""
    models = set()

    # Look for all result files
    for pattern in ["*.json"]:
        for filename in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            basename = os.path.basename(filename)
            # Extract model name from filename (e.g., "mistral-7b_comprehensive_results.json" -> "mistral-7b")
            model_name = basename.split('_')[0]
            models.add(model_name)

    return {"models": sorted(list(models))}


@router.get("/api/visualization/{model_name}/{viz_type}")
async def get_visualization(model_name: str, viz_type: str):
    """Get pre-generated visualization images"""
    viz_files = {
        "accuracy_bar": "accuracy_bar_chart.png",
        "hallucination_rate_bar": "hallucination_rate_bar_chart.png",
        "confidence_bar": "confidence_bar_chart.png",
        "radar": "radar_chart.png",
        "comparison_table": "comparison_table.html"
    }

    if viz_type not in viz_files:
        raise HTTPException(status_code=404, detail=f"Visualization type {viz_type} not found")

    file_path = os.path.join(VISUALIZATION_DIR, viz_files[viz_type])

    if not os.path.exists(file_path):
        # Try to find model-specific visualizations
        model_viz_path = os.path.join(VISUALIZATION_DIR, f"{model_name}_{viz_files[viz_type]}")
        if os.path.exists(model_viz_path):
            file_path = model_viz_path
        else:
            raise HTTPException(status_code=404, detail=f"Visualization file not found")

    if viz_type == "comparison_table":
        with open(file_path, "r") as f:
            content = f.read()
        return {"html": content}
    else:
        return FileResponse(file_path)


@router.post("/api/upload-results")
async def upload_evaluation_results():
    """Endpoint to handle uploaded evaluation results"""
    # This would handle file uploads in a real implementation
    return {"message": "Upload endpoint ready - implement file handling as needed"}


@router.get("/api/model-metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get just the metrics for a model"""
    try:
        results = await get_evaluation_results(model_name)
        return {
            "model": model_name,
            "metrics": results.get("evaluation_results", {}).get("metrics", {}),
            "timestamp": results.get("evaluation_date", datetime.now().isoformat())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/compare-models")
async def compare_models(models: List[str]):
    """Compare multiple models using pre-generated results"""
    comparison_data = {}

    for model_name in models:
        try:
            metrics = await get_model_metrics(model_name)
            comparison_data[model_name] = metrics
        except Exception as e:
            comparison_data[model_name] = {"error": str(e)}

    # Generate comparative analysis
    comparison_metrics = generate_comparison_metrics(comparison_data)

    return {
        "models": comparison_data,
        "comparison": comparison_metrics
    }


def generate_comparison_metrics(models_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparative metrics between models"""
    comparison = {}
    metrics_to_compare = ["accuracy", "hallucination_rate", "confidence"]

    for metric in metrics_to_compare:
        values = {}

        for model_name, data in models_data.items():
            if "metrics" in data and metric in data["metrics"]:
                values[model_name] = data["metrics"][metric]

        if values:
            comparison[metric] = {
                "best_model": max(values.items(), key=lambda x: x[1])[0] if metric != "hallucination_rate" else
                min(values.items(), key=lambda x: x[1])[0],
                "worst_model": min(values.items(), key=lambda x: x[1])[0] if metric != "hallucination_rate" else
                max(values.items(), key=lambda x: x[1])[0],
                "range": max(values.values()) - min(values.values()),
                "average": sum(values.values()) / len(values),
                "values": values
            }

    return comparison
