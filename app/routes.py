from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from app.api_models import ModelClient
from app.config import settings
from app.retrievers import load_test_prompts, MedicalModelEvaluator, load_baseline

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.post("/api/evaluate/medical")
async def evaluate_medical_model(
    model_name: str,
    dataset: str = "pubmed_qa",
    n_samples: int = 10,
    mitigation: str = None,  # "rag", "lora", "ensemble"
):
    n_samples = n_samples or settings.datasets.max_samples
    # Load medical prompts
    prompts = load_test_prompts(dataset, n_samples)

    # Initialize model with selected mitigation
    client = ModelClient.get_client(model_name, mitigation=mitigation)

    # Evaluate with medical-specific metrics
    evaluator = MedicalModelEvaluator()
    results = evaluator.evaluate(client, prompts)

    # Calculate hallucination reduction
    baseline = load_baseline(model_name, dataset)
    reduction = (
        baseline["hallucination_rate"] - results["hallucination_rate"]
    ) / baseline["hallucination_rate"]

    return {
        "results": results,
        "reduction_pct": reduction * 100,
        "mitigation": mitigation,
    }


@router.get("/evaluate", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    return templates.TemplateResponse("evaluation.html", {"request": request})


@router.post("/api/compare")
async def api_compare_models(
    model_names: list[str], dataset: str = "truthful_qa", n_samples: int = 5
):
    prompts = load_test_prompts(dataset, n_samples)
    results = compare_models(model_names, prompts)
    return results


@router.get("/compare", response_class=HTMLResponse)
async def comparison_page(request: Request):
    return templates.TemplateResponse("comparison.html", {"request": request})
