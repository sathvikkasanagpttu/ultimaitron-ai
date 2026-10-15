from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.rag_service import rag_store
from backend.schemas import DocumentUploadResponse, DocumentItem
from typing import List

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
@router.post("", response_model=DocumentUploadResponse, include_in_schema=False)
@router.post("/", response_model=DocumentUploadResponse, include_in_schema=False)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    success, msg, chunks_count, total_chars = rag_store.ingest_document(filename, content_bytes)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return DocumentUploadResponse(
        status="ok",
        filename=filename,
        chunks_count=chunks_count,
        total_characters=total_chars,
        message=f"Indexed {chunks_count} chunks ({total_chars:,} characters)."
    )

@router.get("", response_model=List[DocumentItem])
@router.get("/", response_model=List[DocumentItem], include_in_schema=False)
async def list_documents():
    return rag_store.get_document_list()

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    removed = rag_store.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "ok", "message": f"Document {doc_id} deleted."}

@router.delete("", include_in_schema=False)
@router.delete("/", include_in_schema=False)
@router.post("/clear")
async def clear_documents():
    rag_store.clear()
    return {"status": "ok", "message": "All documents removed."}
