from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routes import api, comparisons, visualizations

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(api.router)
app.include_router(comparisons.router)
app.include_router(visualizations.router)

# Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})
