import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.routers import chat, documents, system

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ultimaitron.core")

# Simple IP-based in-memory rate limiter with window tracking
_request_counts = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=======================================================")
    logger.info("ULTIMAITRON AI 2.0 — Canonical Core Gateway starting")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Active LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Allowed CORS Origins: {settings.allowed_origins}")
    logger.info("=======================================================")
    yield
    logger.info("ULTIMAITRON AI 2.0 — Canonical Core Gateway shutting down.")


app = FastAPI(
    title="ULTIMAITRON AI 2.0 Core API",
    description="Full-Stack Holographic AI Companion Brain & Assistant Backend",
    version="2.0.0",
    lifespan=lifespan
)

# Enforce secure CORS policy: explicit origins, never wildcard with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Security Headers & Rate Limiting Middleware
@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    # 1. Rate limiting on /api endpoints
    if request.url.path.startswith("/api/chat") or request.url.path.startswith("/api/documents"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - settings.rate_limit_window_seconds

        # Prune older entries
        _request_counts[client_ip] = [t for t in _request_counts[client_ip] if t > window_start]
        if len(_request_counts[client_ip]) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a moment before trying again."}
            )
        _request_counts[client_ip].append(now)

    response: Response = await call_next(request)

    # 2. Hardened security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response


# Mount unified canonical API routers
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
