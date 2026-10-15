from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=40)
    document_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("messages")
    @classmethod
    def conversation_ends_with_user(cls, value: list[Message]) -> list[Message]:
        if value[-1].role != "user":
            raise ValueError("The final message must be from the user")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider_configured: bool
    model: str


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    status: Literal["ready"] = "ready"


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentClearResponse(BaseModel):
    removed: int
