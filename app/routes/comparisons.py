from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models.evaluation import compare_models
from app.utils.dataset_loader import load_test_prompts

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


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
