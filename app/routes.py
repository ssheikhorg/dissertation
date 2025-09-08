import glob
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from utils import generate_comparison_metrics


class ModelsEnum:
    """Enum for supported models"""

    LLAMA_2_7B = "llama-2-7b"
    MISTRAL_7B = "mistral-7b"
    QWEN_7B = "qwen-7b"
    MEDITRON_7B = "meditron-7b"
    BIOMEDGPT = "biomedgpt"
    GPT_4 = "gpt-4"
    CLAUDE_3_OPUS = "claude-3-opus"
    GROK = "grok"


router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Directory where evaluation results are stored
RESULTS_DIR = "app/evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory for pre-generated visualizations
VISUALIZATION_DIR = "app/visualizations"
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

        with open(latest_file) as f:
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
            model_name = basename.split("_")[0]
            models.add(model_name)

    return {"models": sorted(list(models))}


@router.get("/api/visualization/{viz_type}")
async def get_visualization(viz_type: str):
    """Get pre-generated visualization images"""
    viz_mapping = {
        "accuracy_bar": "accuracy_bar_chart.png",
        "hallucination_rate_bar": "hallucination_rate_bar_chart.png",
        "confidence_bar": "confidence_bar_chart.png",
        "radar": "radar_chart.png",
        "comparison_table": "comparison_table.html",
    }

    if viz_type not in viz_mapping:
        raise HTTPException(status_code=404, detail=f"Visualization type {viz_type} not supported")

    file_name = viz_mapping[viz_type]
    file_path = os.path.join(VISUALIZATION_DIR, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Visualization file {file_name} not found")

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

        return {"model": model_name, "metrics": metrics}
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
