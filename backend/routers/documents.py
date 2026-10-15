import time
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from backend.config import settings
from backend.schemas import DocumentClearResponse, DocumentInfo, DocumentItem, DocumentListResponse, DocumentUploadResponse
from backend.services.rag_service import rag_store

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
@router.post("", response_model=DocumentUploadResponse, status_code=201, include_in_schema=False)
@router.post("/", response_model=DocumentUploadResponse, status_code=201, include_in_schema=False)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    max_bytes = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024

    try:
        content_bytes = await file.read(max_bytes + 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(content_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {settings.MAX_DOCUMENT_SIZE_MB}MB.")

    success, msg, chunks_count, total_chars = rag_store.ingest_document(filename, content_bytes)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    # Find the newly added document id
    doc_id = list(rag_store.documents.keys())[-1] if rag_store.documents else "doc"

    return DocumentUploadResponse(
        status="ok",
        filename=filename,
        chunks_count=chunks_count,
        total_characters=total_chars,
        message=f"Indexed {chunks_count} chunks ({total_chars:,} characters).",
        id=doc_id
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents():
    items = rag_store.get_document_list()
    doc_infos = [
        DocumentInfo(id=item.id, name=item.filename, chunks=item.chunks_count, status="ready")
        for item in items
    ]
    return DocumentListResponse(documents=doc_infos)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    removed = rag_store.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document was not found or has already been removed.")
    return None


@router.delete("", response_model=DocumentClearResponse)
@router.delete("/", response_model=DocumentClearResponse, include_in_schema=False)
@router.post("/clear", response_model=DocumentClearResponse)
async def clear_documents():
    count = rag_store.clear()
    return DocumentClearResponse(removed=count)
