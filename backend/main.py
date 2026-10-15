import time
import os
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.routers import chat, documents, system

# Simple IP-based in-memory rate limiter
_request_counts = defaultdict(list)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[ULTIMAITRON AI 2.0] Backend starting on {settings.HOST}:{settings.PORT}")
    print(f"[ULTIMAITRON AI 2.0] LLM Provider active: {settings.LLM_PROVIDER}")
    yield
    print("[ULTIMAITRON AI 2.0] Backend shutting down.")

app = FastAPI(
    title="ULTIMAITRON AI 2.0 Core API",
    description="Full-Stack Holographic AI Companion Brain & Assistant Backend",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows development and configured origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit API calls
    if request.url.path.startswith("/api/chat"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        minute_ago = now - 60
        
        # Clean old timestamps
        _request_counts[client_ip] = [t for t in _request_counts[client_ip] if t > minute_ago]
        if len(_request_counts[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please wait a moment."}
            )
        _request_counts[client_ip].append(now)

    response = await call_next(request)
    return response

# Include routers
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(system.router)

# Mount static dist/ if it exists (for standalone production preview)
dist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist")
if os.path.isdir(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
