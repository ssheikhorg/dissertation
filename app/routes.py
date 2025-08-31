import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .clients import ModelClient, generate_visualization_data
from .data import load_baseline, load_test_prompts
from .evaluators import (
    MedicalModelEvaluator,
    compare_models,
    generate_improvement_suggestions,
    prepare_bar_chart_data,
    prepare_radar_data,
    prepare_scatter_data,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/api/evaluate")
async def evaluate_model(
    model_name: str = Form(...),
    dataset: str = Form("pubmed_qa"),
    sample_count: int = Form(5),
    mitigation: str = Form(None),
):
    """Evaluate a single model with focus on hallucination metrics"""
    try:
        # Load prompts
        prompts = load_test_prompts(dataset, min(sample_count, 100))  # Fixed max_samples

        # Initialize model client with mitigation
        client = ModelClient.get_client(model_name, mitigation=mitigation)

        # Evaluate with medical-specific metrics
        evaluator = MedicalModelEvaluator()
        results = evaluator.evaluate(client, prompts)

        # Get baseline for comparison
        baseline = load_baseline(model_name, dataset)

        # Calculate improvement metrics
        hallucination_reduction = (
            ((baseline["hallucination_rate"] - results["hallucination_rate"]) / baseline["hallucination_rate"] * 100)
            if baseline["hallucination_rate"] > 0
            else 0
        )

        # Prepare response data
        response_data = {
            "model": model_name,
            "dataset": dataset,
            "sample_count": sample_count,
            "metrics": results,
            "baseline": baseline,
            "improvement": {
                "hallucination_reduction": hallucination_reduction,
                "accuracy_improvement": ((results["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100)
                if baseline["accuracy"] > 0
                else 0,
            },
            "suggestions": generate_improvement_suggestions(results, baseline),
            "sample_responses": results.get("sample_responses", [])[:3],
        }

        # Save evaluation result to file
        eval_dir = "evaluations"
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset}_{timestamp}.json"
        filepath = os.path.join(eval_dir, filename)

        with open(filepath, "w") as f:
            json.dump(response_data, f, indent=2)

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/compare")
async def api_compare_models(
    model1: str = Form(...),
    model2: str = Form(...),
    dataset: str = Form("pubmed_qa"),
    sample_count: int = Form(5),
    mitigation: str = Form(None),
):
    """Compare two models with focus on hallucination metrics"""
    try:
        prompts = load_test_prompts(dataset, min(sample_count, 100))

        # Evaluate both models using the compare_models function
        comparison_results = await compare_models([model1, model2], prompts, mitigation)

        # Prepare comparison data
        comparison_data = {
            "models": comparison_results["models"],
            "comparison": comparison_results["comparison"],
            "prompt_count": len(prompts),
            "dataset_stats": comparison_results.get("dataset_stats", {}),
        }

        # Save comparison result to file
        eval_dir = "evaluations"
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{model1}_vs_{model2}_{dataset}_{timestamp}.json"
        filepath = os.path.join(eval_dir, filename)

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2)

        return comparison_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/visualize")
async def generate_visualization(
    visualization_type: str,
    metric: str,
    models: list[str] = None,
):
    """Generate visualization data for the dashboard"""
    try:
        if models is None:
            models = list(ModelClient._registry.keys())

        # Generate visualization data
        general_df, bias_df = await generate_visualization_data(
            model_names=models,
            metrics=[metric],
            n_samples=50,
        )

        # Prepare data based on visualization type
        if visualization_type == "bar":
            chart_data = prepare_bar_chart_data(general_df, metric, models)
        elif visualization_type == "radar":
            chart_data = prepare_radar_data(general_df, models, [metric])
        elif visualization_type == "scatter":
            secondary_metric = "accuracy" if "hallucination" in metric else "hallucination_rate"
            chart_data = prepare_scatter_data(general_df, metric, secondary_metric)
        else:
            return {"error": "Unsupported visualization type"}

        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/recent-evaluations")
async def get_recent_evaluations():
    """Get recently completed evaluations for the dashboard"""
    try:
        eval_dir = "evaluations"
        os.makedirs(eval_dir, exist_ok=True)

        recent_evaluations = []

        # Look for evaluation files from last 7 days
        for filename in os.listdir(eval_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(eval_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))

                if datetime.now() - file_time < timedelta(days=7):
                    with open(filepath) as f:
                        eval_data = json.load(f)
                        recent_evaluations.append(
                            {
                                "model": eval_data.get("model", "Unknown"),
                                "dataset": eval_data.get("dataset", "Unknown"),
                                "timestamp": file_time.isoformat(),
                                "hallucination_rate": eval_data.get("metrics", {}).get("hallucination_rate", 0),
                                "accuracy": eval_data.get("metrics", {}).get("accuracy", 0),
                            }
                        )

        # Sort by most recent
        recent_evaluations.sort(key=lambda x: x["timestamp"], reverse=True)

        return recent_evaluations[:5]  # Return top 5 most recent

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
