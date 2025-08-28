from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes import router

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(router)

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})
