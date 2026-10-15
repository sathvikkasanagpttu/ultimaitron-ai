import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from backend.config import settings
from backend.schemas import ChatMessage
from backend.services.llm_service import stream_chat_response
from backend.services.rag_service import rag_store

router = APIRouter(prefix="/api/chat", tags=["chat"])

# In-memory session history
_conversation_history: List[ChatMessage] = []


@router.post("", include_in_schema=True)
@router.post("/", include_in_schema=False)
@router.post("/stream", include_in_schema=True)
async def chat_stream(request: Request):
    """
    Server-Sent Events endpoint streaming AI tokens, tool calls, and states.
    Accepts both:
    1. ULTIMAITRON 2.0 schema: {"message": "...", "history": [...]}
    2. Legacy schema: {"messages": [{"role": "user", "content": "..."}], "document_ids": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    message_text = ""
    history_list: List[ChatMessage] = []
    document_ids: List[str] = body.get("document_ids", [])

    if "message" in body:
        message_text = str(body["message"]).strip()
        raw_history = body.get("history", [])
        if isinstance(raw_history, list):
            for h in raw_history:
                if isinstance(h, dict) and "content" in h:
                    history_list.append(ChatMessage(role=h.get("role", "user"), content=str(h["content"])))
    elif "messages" in body and isinstance(body["messages"], list) and len(body["messages"]) > 0:
        msgs = body["messages"]
        last_msg = msgs[-1]
        message_text = str(last_msg.get("content", "")).strip()
        for m in msgs[:-1]:
            if isinstance(m, dict) and "content" in m:
                history_list.append(ChatMessage(role=m.get("role", "user"), content=str(m["content"])))

    if not message_text:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    # If document_ids were explicitly passed, verify they exist
    if document_ids and not rag_store.contains(document_ids):
        raise HTTPException(status_code=410, detail="An uploaded document expired after a backend restart. Upload it again to continue.")

    # Check provider configuration if external provider is explicitly configured
    active_prov = (body.get("provider") or settings.LLM_PROVIDER or "").lower()
    if active_prov == "anthropic" and not settings.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Anthropic API key is not configured. Set ANTHROPIC_API_KEY on the backend.")
    elif active_prov == "openai" and not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured. Set OPENAI_API_KEY or LLM_API_KEY on the backend.")

    user_msg = ChatMessage(role="user", content=message_text)
    _conversation_history.append(user_msg)

    effective_history = history_list if history_list else _conversation_history[:-1]
    provider = body.get("provider", None)
    model = body.get("model", None)
    try:
        temperature = float(body.get("temperature", 0.7))
    except (ValueError, TypeError):
        temperature = 0.7
    system_prompt = body.get("system_prompt", None)

    async def event_generator():
        async for chunk in stream_chat_response(
            message=message_text,
            history=effective_history,
            provider=provider,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            document_ids=document_ids
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history")
async def get_history():
    return {"history": _conversation_history}


@router.post("/clear")
async def clear_history():
    _conversation_history.clear()
    return {"status": "cleared", "message": "Conversation history erased."}
