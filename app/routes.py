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
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory for pre-generated visualizations
VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


@router.get("/api/evaluation/{model_name}")
async def get_evaluation_results(model_name: str):
    """Get evaluation results for a specific model from downloaded files"""
    try:
        # Look for the comprehensive results file in model-specific subdirectory
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"No evaluation directory found for {model_name}")

        # Look for comprehensive results file
        pattern = os.path.join(model_dir, f"{model_name}_comprehensive_results.json")
        files = glob.glob(pattern)

        if not files:
            # Fallback to UI export data
            pattern = os.path.join(model_dir, f"{model_name}_ui_export_data.json")
            files = glob.glob(pattern)

        if not files:
            # Fallback: look for any JSON file in the model directory
            pattern = os.path.join(model_dir, "*.json")
            files = glob.glob(pattern)

        if not files:
            raise HTTPException(status_code=404, detail=f"No evaluation results found for {model_name}")

        # Use the most recent file
        latest_file = max(files, key=os.path.getctime)

        with open(latest_file) as f:
            results = json.load(f)

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")

@router.get("/api/models/available")
async def get_available_models():
    """Get list of models that have evaluation results"""
    models = set()

    # Look for all model directories
    if not os.path.exists(RESULTS_DIR):
        return {"models": []}

    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if os.path.isdir(item_path):
            # Check if this directory has any JSON files
            json_files = glob.glob(os.path.join(item_path, "*.json"))
            if json_files:
                models.add(item)

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
    }

    if viz_type not in viz_mapping:
        raise HTTPException(status_code=404, detail=f"Visualization type {viz_type} not supported")

    file_name = viz_mapping[viz_type]
    file_path = os.path.join(VISUALIZATION_DIR, model_name, file_name)

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


async def generate_fallback_visualization(model_name: str, viz_type: str):
    """Generate a simple visualization if pre-generated file is missing"""
    try:
        # Get model data to generate a chart
        results = await get_evaluation_results(model_name)
        metrics = results.get('metrics', {}) or results.get('evaluation_results', {}).get('metrics', {})

        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics available for visualization")

        # For now, just return a message that we need to generate visualizations
        # In a real implementation, you could generate charts dynamically here
        raise HTTPException(
            status_code=404,
            detail=f"Pre-generated visualization not found. Please generate visualizations for {model_name} first."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")
