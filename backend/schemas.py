import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
    timestamp: float = Field(default_factory=time.time)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class ChatRequest(BaseModel):
    # Support both single-message and multi-turn payloads
    message: Optional[str] = Field(None, min_length=1, max_length=12000, description="The user's input prompt")
    messages: Optional[List[ChatMessage]] = Field(None, description="Prior conversation message array")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversation history")
    document_ids: List[str] = Field(default_factory=list, description="IDs of documents to ground response in")
    provider: Optional[str] = Field(None, description="Override LLM provider (anthropic, openai, ollama, gemini, mock)")
    model: Optional[str] = Field(None, description="Override LLM model name")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: Optional[str] = Field(None, description="Optional system instructions")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Message cannot be blank")
        return v


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]


class ChatStreamEvent(BaseModel):
    type: Literal["token", "state", "tool_call", "citation", "sources", "done", "error"]
    data: Any


class DocumentChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    page: Optional[int] = None


class DocumentItem(BaseModel):
    id: str
    filename: str
    chunks_count: int
    total_characters: int
    uploaded_at: float
    status: Literal["ready", "processing", "error"] = "ready"

    # Compatibility alias
    @property
    def name(self) -> str:
        return self.filename

    @property
    def chunks(self) -> int:
        return self.chunks_count


class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks_count: int
    total_characters: int
    message: str
    id: Optional[str] = None

    # Compatibility properties for legacy test suite
    @property
    def name(self) -> str:
        return self.filename

    @property
    def chunks(self) -> int:
        return self.chunks_count


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    status: Literal["ready"] = "ready"


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentClearResponse(BaseModel):
    removed: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    provider: str = "mock"
    provider_configured: bool = False
    model: str = "auto"
    uptime_seconds: float = 0.0
    total_conversations: int = 0
    documents_count: int = 0
