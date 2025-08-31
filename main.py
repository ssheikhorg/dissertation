import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import router

app = FastAPI()

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("evaluations", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    # Respect an env toggle for reload, but force it off under PyCharm on Windows to avoid multiprocessing issues
    reload_flag = os.environ.get("UVICORN_RELOAD", "").lower() in {"1", "true", "yes"}
    if os.name == "nt" and os.environ.get("PYCHARM_HOSTED"):
        reload_flag = False

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=reload_flag,
    )
