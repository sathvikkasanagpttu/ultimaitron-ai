import uvicorn

from .config import get_settings

if __name__ == "__main__":
    config = get_settings()
    uvicorn.run("backend.app.main:app", host=config.backend_host, port=config.backend_port, reload=False)
