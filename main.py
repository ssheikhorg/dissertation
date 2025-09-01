import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import router

app = FastAPI()

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/baselines", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("evaluations", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    reload_flag = os.environ.get("UVICORN_RELOAD", "").lower() in {"1", "true", "yes"}
    if os.name == "nt" and os.environ.get("PYCHARM_HOSTED"):
        reload_flag = False

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=reload_flag,
    )
