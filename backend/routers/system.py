import time
from fastapi import APIRouter
from backend.config import settings
from backend.schemas import HealthResponse
from backend.services.command_registry import get_tool_definitions
from backend.services.llm_service import detect_provider
from backend.services.rag_service import rag_store

router = APIRouter(prefix="/api", tags=["system"])
_start_time = time.time()
_total_requests = 0


@router.get("/health", response_model=HealthResponse)
async def health():
    active_prov = detect_provider()
    configured = (
        bool(settings.ANTHROPIC_API_KEY) if active_prov == "anthropic"
        else bool(settings.OPENAI_API_KEY) if active_prov == "openai"
        else True if active_prov in ["mock", "ollama"]
        else False
    )
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        provider=active_prov,
        provider_configured=configured,
        model=settings.LLM_MODEL or settings.openai_model or "auto",
        uptime_seconds=round(time.time() - _start_time, 2),
        total_conversations=_total_requests,
        documents_count=len(rag_store.documents)
    )


@router.get("/ready")
async def readiness():
    return {
        "status": "ready",
        "service": "ultimaitron-core",
        "version": "2.0.0",
        "storage": "in-memory-rag",
        "llm_provider": detect_provider()
    }


@router.get("/commands")
async def list_commands():
    return {"commands": get_tool_definitions()}
