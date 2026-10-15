from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: float = Field(default_factory=time.time)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="The user's input prompt")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversation history")
    provider: Optional[str] = Field(None, description="Override LLM provider (anthropic, openai, ollama, gemini, mock)")
    model: Optional[str] = Field(None, description="Override LLM model name")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: Optional[str] = Field(None, description="Optional system instructions")

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

class ChatStreamEvent(BaseModel):
    type: Literal["token", "state", "tool_call", "citation", "done", "error"]
    data: Any

class DocumentChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str

class DocumentItem(BaseModel):
    id: str
    filename: str
    chunks_count: int
    total_characters: int
    uploaded_at: float

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks_count: int
    total_characters: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    provider: str
    model: str
    uptime_seconds: float
    total_conversations: int
    documents_count: int
