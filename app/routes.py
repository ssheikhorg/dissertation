# routes.py - Simplified routes for hallucination evaluation
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List

from .clients import ModelFactory
from .evaluators import HallucinationEvaluator, generate_improvement_suggestions
from .data import load_test_prompts, load_baseline

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/api/evaluate")
async def evaluate_model(
        model_name: str = Form(...),
        dataset: str = Form("pubmed_qa"),
        sample_count: int = Form(5)
):
    """Evaluate a single model for hallucinations"""
    try:
        # Get model client
        client = ModelFactory.get_client(model_name)

        # Load test prompts
        prompts = load_test_prompts(dataset, min(sample_count, 10))  # Reduced for local models

        # Evaluate with hallucination-focused metrics
        evaluator = HallucinationEvaluator()
        results = await evaluator.evaluate_model(client, prompts, dataset, sample_count)

        # Get baseline for comparison
        baseline = load_baseline(model_name, dataset)

        # Calculate improvement metrics
        hallucination_reduction = (
            ((baseline["hallucination_rate"] - results["metrics"]["hallucination_rate"]) / baseline[
                "hallucination_rate"] * 100)
            if baseline["hallucination_rate"] > 0
            else 0
        )

        # Prepare response data
        response_data = {
            "model": model_name,
            "dataset": dataset,
            "sample_count": sample_count,
            "metrics": results["metrics"],
            "baseline": baseline,
            "improvement": {
                "hallucination_reduction": hallucination_reduction,
                "accuracy_improvement": (
                            (results["metrics"]["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100)
                if baseline["accuracy"] > 0
                else 0,
            },
            "suggestions": generate_improvement_suggestions(results["metrics"]),
            "sample_responses": results.get("sample_responses", [])[:3],
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/compare")
async def compare_models(
        model1: str = Form(...),
        model2: str = Form(...),
        dataset: str = Form("pubmed_qa"),
        sample_count: int = Form(5)
):
    """Compare two models for hallucinations"""
    try:
        prompts = load_test_prompts(dataset, min(sample_count, 10))

        all_results = {}
        evaluator = HallucinationEvaluator()

        for model_name in [model1, model2]:
            try:
                client = ModelFactory.get_client(model_name)
                results = await evaluator.evaluate_model(client, prompts, dataset, sample_count)

                all_results[model_name] = {
                    "metrics": results["metrics"],
                    "sample_responses": results.get("sample_responses", [])[:2],
                }
            except Exception as e:
                all_results[model_name] = {"error": str(e), "metrics": None}

        # Generate comparative analysis
        comparison = generate_comparison_metrics(all_results)

        comparison_data = {
            "models": all_results,
            "comparison": comparison,
            "prompt_count": len(prompts),
            "dataset_stats": get_dataset_stats(prompts),
        }

        return comparison_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/visualize")
async def generate_visualization(
        visualization_type: str,
        metric: str,
        models: List[str] = None
):
    """Generate visualization data for hallucination metrics"""
    try:
        if models is None:
            models = ["llama-2-7b", "mistral-7b", "qwen-7b"]

        # Generate visualization data
        chart_data = prepare_chart_data(visualization_type, metric, models)

        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models")
async def get_available_models():
    """Get list of available local models"""
    try:
        models = ModelFactory.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Helper functions
def generate_comparison_metrics(all_results: dict) -> dict:
    """Generate comparative metrics between models"""
    comparison = {}
    metrics = ["accuracy", "hallucination_rate", "confidence"]

    for metric in metrics:
        values = {
            model: results["metrics"][metric]
            for model, results in all_results.items()
            if results.get("metrics")
        }

        if values:
            comparison[metric] = {
                "best_model": max(values.items(), key=lambda x: x[1])[0] if metric == "accuracy" else
                min(values.items(), key=lambda x: x[1])[0],
                "worst_model": min(values.items(), key=lambda x: x[1])[0] if metric == "accuracy" else
                max(values.items(), key=lambda x: x[1])[0],
                "range": max(values.values()) - min(values.values()),
                "mean": sum(values.values()) / len(values),
                "values": values,
            }

    return comparison


def get_dataset_stats(prompts: list) -> dict:
    """Calculate basic statistics about the evaluation dataset"""
    ref_lengths = [len(p["original_reference"]) for p in prompts]
    prompt_lengths = [len(p["original_prompt"]) for p in prompts]

    return {
        "prompt_length": {
            "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "min": min(prompt_lengths) if prompt_lengths else 0,
        },
        "reference_length": {
            "mean": sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0,
            "max": max(ref_lengths) if ref_lengths else 0,
            "min": min(ref_lengths) if ref_lengths else 0,
        },
        "total_prompts": len(prompts),
    }


def prepare_chart_data(visualization_type: str, metric: str, models: List[str]) -> dict:
    """Prepare chart data for visualization"""
    # Demo data - in real implementation, this would come from actual evaluations
    demo_data = {
        "llama-2-7b": {"accuracy": 0.72, "hallucination_rate": 0.18, "confidence": 0.75},
        "mistral-7b": {"accuracy": 0.68, "hallucination_rate": 0.22, "confidence": 0.70},
        "qwen-7b": {"accuracy": 0.75, "hallucination_rate": 0.15, "confidence": 0.78},
        "meditron-7b": {"accuracy": 0.78, "hallucination_rate": 0.12, "confidence": 0.82},
        "biomedgpt": {"accuracy": 0.82, "hallucination_rate": 0.08, "confidence": 0.88}
    }

    if visualization_type == "bar":
        return {
            "type": "bar",
            "data": {
                "labels": models,
                "datasets": [{
                    "label": metric.replace("_", " ").title(),
                    "data": [demo_data[model].get(metric, 0) for model in models],
                    "backgroundColor": ["rgba(54, 162, 235, 0.6)" for _ in models],
                    "borderColor": ["rgba(54, 162, 235, 1)" for _ in models],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{metric.replace('_', ' ').title()} Comparison",
                        "font": {"size": 16}
                    }
                },
                "scales": {
                    "y": {"beginAtZero": True, "max": 1}
                }
            }
        }
    else:
        return {"error": "Unsupported visualization type"}