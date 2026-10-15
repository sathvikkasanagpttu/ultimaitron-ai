from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas import ChatRequest, ChatMessage
from backend.services.llm_service import stream_chat_response
from typing import List

router = APIRouter(prefix="/api/chat", tags=["chat"])

# In-memory session history
_conversation_history: List[ChatMessage] = []

@router.post("")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events endpoint streaming AI tokens, tool calls, and states.
    """
    # Append user message
    user_msg = ChatMessage(role="user", content=request.message)
    _conversation_history.append(user_msg)
    
    # Use request.history if supplied, otherwise server session history
    effective_history = request.history if request.history else _conversation_history[:-1]

    async def event_generator():
        full_assistant_reply = []
        async for chunk in stream_chat_response(
            message=request.message,
            history=effective_history,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            system_prompt=request.system_prompt
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
