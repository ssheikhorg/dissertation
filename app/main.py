from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes import router

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(router)

# Templates
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})
