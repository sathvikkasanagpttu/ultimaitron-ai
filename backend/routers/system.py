import time
from fastapi import APIRouter
from backend.config import settings
from backend.schemas import HealthResponse
from backend.services.command_registry import get_tool_definitions
from backend.services.rag_service import rag_store
from backend.services.llm_service import detect_provider

router = APIRouter(prefix="/api", tags=["system"])
_start_time = time.time()
_total_requests = 0

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        provider=detect_provider(),
        model=settings.LLM_MODEL or "auto",
        uptime_seconds=round(time.time() - _start_time, 2),
        total_conversations=_total_requests,
        documents_count=len(rag_store.documents)
    )

@router.get("/commands")
async def list_commands():
    return {"commands": get_tool_definitions()}
