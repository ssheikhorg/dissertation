from fastapi import APIRouter, Form, HTTPException
from starlette.templating import Jinja2Templates

from .clients import ModelClient, generate_visualization_data
from .config import settings
from .data import load_baseline, load_test_prompts
from .evaluators import (
    MedicalModelEvaluator,
    generate_improvement_suggestions,
    prepare_bar_chart_data,
    prepare_radar_data,
    prepare_scatter_data,
)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


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
        prompts = load_test_prompts(dataset, min(sample_count, settings.datasets.max_samples))

        # Initialize model client
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

        # Prepare response for HTML dashboard
        return {
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/compare")
async def api_compare_models(
    model1: str = Form(...),
    model2: str = Form(...),
    dataset: str = Form("pubmed_qa"),
    sample_count: int = Form(5),
):
    """Compare two models with focus on hallucination metrics"""
    try:
        prompts = load_test_prompts(dataset, min(sample_count, settings.datasets.max_samples))

        # Evaluate both models
        evaluator = MedicalModelEvaluator()

        # Evaluate first model
        client1 = ModelClient.get_client(model1)
        results1 = evaluator.evaluate(client1, prompts)

        # Evaluate second model
        client2 = ModelClient.get_client(model2)
        results2 = evaluator.evaluate(client2, prompts)

        # Prepare comparison data
        comparison_data = {
            "models": {
                model1: results1,
                model2: results2,
            },
            "comparison": {
                "hallucination_difference": results1["hallucination_rate"] - results2["hallucination_rate"],
                "accuracy_difference": results1["accuracy"] - results2["accuracy"],
                "better_model": (
                    model1 if results1["hallucination_rate"] < results2["hallucination_rate"] else model2
                ),
            },
        }

        return comparison_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            n_samples=50,  # Smaller sample for faster visualization
        )

        # Prepare data based on visualization type
        if visualization_type == "bar":
            # Prepare data for bar chart
            chart_data = prepare_bar_chart_data(general_df, metric)
        elif visualization_type == "radar":
            # Prepare data for radar chart
            chart_data = prepare_radar_data(general_df, models, [metric])
        elif visualization_type == "scatter":
            # Prepare data for scatter plot
            chart_data = prepare_scatter_data(general_df, metric)
        else:
            return {"error": "Unsupported visualization type"}

        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
