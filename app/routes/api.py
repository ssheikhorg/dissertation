from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from app.models.evaluation import evaluate_model
from app.utils.dataset_loader import load_test_prompts

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.post("/api/evaluate")
async def api_evaluate_model(
    model_name: str, dataset: str = "truthful_qa", n_samples: int = 10
):
    prompts = load_test_prompts(dataset, n_samples)
    results = evaluate_model(model_name, prompts)
    return results


@router.get("/evaluate", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    return templates.TemplateResponse("evaluation.html", {"request": request})
