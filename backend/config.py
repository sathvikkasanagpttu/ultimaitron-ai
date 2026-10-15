from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", validation_alias="BACKEND_HOST")
    port: int = Field(default=8000, validation_alias="BACKEND_PORT", ge=1, le=65535)

    # LLM Provider: 'auto', 'anthropic', 'openai', 'gemini', 'ollama', or 'mock'
    llm_provider: str = Field(default="mock", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")

    # Provider Keys & Endpoints
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", validation_alias="ANTHROPIC_MODEL")

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL")

    # Legacy support aliases
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", validation_alias="LLM_BASE_URL")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")

    # Generation & Timeout parameters
    max_tokens: int = Field(default=1024, validation_alias="MAX_TOKENS", ge=64, le=8192)
    temperature: float = Field(default=0.7, validation_alias="TEMPERATURE", ge=0.0, le=2.0)
    request_timeout: float = Field(default=60.0, validation_alias="LLM_TIMEOUT_SECONDS", ge=5.0, le=300.0)

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_REQUESTS", ge=1, le=1000)
    rate_limit_window_seconds: int = Field(default=60, validation_alias="RATE_LIMIT_WINDOW_SECONDS", ge=1, le=3600)

    # CORS Origins (comma-separated string or list)
    frontend_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,http://localhost:8080,http://127.0.0.1:8080,http://localhost:8000,http://127.0.0.1:8000",
        validation_alias="FRONTEND_ORIGINS"
    )

    # RAG / Storage
    max_document_size_mb: int = Field(default=10, validation_alias="MAX_DOCUMENT_SIZE_MB", ge=1, le=50)
    chunk_size: int = Field(default=500, validation_alias="CHUNK_SIZE", ge=100, le=2000)
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP", ge=0, le=500)
    max_documents: int = Field(default=20, validation_alias="MAX_DOCUMENTS", ge=1, le=100)
    max_pdf_pages: int = Field(default=1000, validation_alias="MAX_PDF_PAGES", ge=1, le=5000)
    max_extracted_chars: int = Field(default=1_000_000, validation_alias="MAX_EXTRACTED_CHARS", ge=10_000, le=10_000_000)

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]

    # Backward compatibility uppercase properties for existing code
    @property
    def HOST(self) -> str:
        return self.host

    @property
    def PORT(self) -> int:
        return self.port

    @property
    def LLM_PROVIDER(self) -> str:
        return self.llm_provider

    @property
    def LLM_MODEL(self) -> str:
        return self.llm_model or self.openai_model or self.anthropic_model

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return self.anthropic_api_key

    @property
    def OPENAI_API_KEY(self) -> str:
        return self.openai_api_key or self.llm_api_key

    @property
    def GEMINI_API_KEY(self) -> str:
        return self.gemini_api_key

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self.ollama_base_url

    @property
    def MAX_TOKENS(self) -> int:
        return self.max_tokens

    @property
    def TEMPERATURE(self) -> float:
        return self.temperature

    @property
    def REQUEST_TIMEOUT(self) -> float:
        return self.request_timeout

    @property
    def RATE_LIMIT_PER_MINUTE(self) -> int:
        return self.rate_limit_per_minute

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return self.allowed_origins

    @property
    def MAX_DOCUMENT_SIZE_MB(self) -> int:
        return self.max_document_size_mb

    @property
    def CHUNK_SIZE(self) -> int:
        return self.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.chunk_overlap


settings = Settings()
