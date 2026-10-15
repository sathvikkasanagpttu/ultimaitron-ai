import io
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pypdf import PdfReader
from backend.config import settings
from backend.schemas import DocumentChunk, DocumentItem


class DocumentError(ValueError):
    pass


def _safe_filename(filename: str) -> str:
    # Defend against path traversal: sanitize directory separators and illegal characters
    basename = filename.replace("\\", "/").split("/")[-1].strip()
    cleaned = re.sub(r"[^\w .()-]", "_", basename, flags=re.UNICODE).strip(" .")
    return (cleaned or "document")[:160]


class DocumentKnowledgeStore:
    def __init__(self):
        self.documents: Dict[str, DocumentItem] = {}
        self.chunks: List[DocumentChunk] = []
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def clear(self) -> int:
        count = len(self.documents)
        self.documents.clear()
        self.chunks.clear()
        self._vocabulary.clear()
        self._idf.clear()
        return count

    def get_document_list(self) -> List[DocumentItem]:
        return list(self.documents.values())

    def contains(self, document_ids: List[str]) -> bool:
        if not document_ids:
            return True
        return all(doc_id in self.documents for doc_id in document_ids)

    def remove_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            self._rebuild_index()
            return True
        return False

    def ingest_document(self, filename: str, content_bytes: bytes) -> Tuple[bool, str, int, int]:
        if len(self.documents) >= settings.max_documents:
            return False, f"Maximum of {settings.max_documents} documents reached. Remove a document before uploading another.", 0, 0

        clean_name = _safe_filename(filename)
        ext = os.path.splitext(clean_name)[1].lower()
        if ext not in [".txt", ".pdf", ".md", ".json"]:
            return False, f"Unsupported file extension '{ext}'. Only .txt, .pdf, .md, .json are supported.", 0, 0

        if not content_bytes:
            return False, "The uploaded file is empty.", 0, 0

        max_bytes = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
        if len(content_bytes) > max_bytes:
            return False, f"File exceeds maximum allowed size of {settings.MAX_DOCUMENT_SIZE_MB}MB.", 0, 0

        pages_extracted: List[Tuple[str, Optional[int]]] = []
        total_extracted_chars = 0

        try:
            if ext == ".pdf":
                if not content_bytes.startswith(b"%PDF-"):
                    return False, "The file does not have a valid PDF signature.", 0, 0

                try:
                    reader = PdfReader(io.BytesIO(content_bytes), strict=False)
                    if len(reader.pages) > settings.max_pdf_pages:
                        return False, f"PDF exceeds the {settings.max_pdf_pages} page processing limit.", 0, 0

                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text() or ""
                        clean_page_text = re.sub(r"\s+", " ", text).strip()
                        total_extracted_chars += len(clean_page_text)
                        if total_extracted_chars > settings.max_extracted_chars:
                            return False, f"Extracted text exceeds the {settings.max_extracted_chars:,} character processing limit.", 0, 0
                        if clean_page_text:
                            pages_extracted.append((clean_page_text, page_num))
                except Exception as e:
                    return False, f"Could not read this PDF. Check that it is not encrypted or damaged: {str(e)}", 0, 0
            else:
                try:
                    raw_text = content_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        raw_text = content_bytes.decode("utf-8", errors="replace")
                    except Exception as e:
                        return False, f"Encoding error: TXT/MD/JSON files must be valid UTF-8. {str(e)}", 0, 0

                clean_text = re.sub(r"\s+", " ", raw_text).strip()
                if len(clean_text) > settings.max_extracted_chars:
                    return False, f"Extracted text exceeds the {settings.max_extracted_chars:,} character processing limit.", 0, 0
                if clean_text:
                    total_extracted_chars += len(clean_text)
                    pages_extracted.append((clean_text, None))

        except Exception as e:
            return False, f"Error processing document: {str(e)}", 0, 0

        if not pages_extracted or total_extracted_chars == 0:
            return False, "No readable text was found in this document.", 0, 0

        doc_id = str(uuid.uuid4())[:8]
        new_chunks: List[DocumentChunk] = []

        chunk_idx = 0
        total_chars = 0
        for page_text, page_num in pages_extracted:
            total_chars += len(page_text)
            page_chunks = self._split_page(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            for chunk_str in page_chunks:
                new_chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    filename=clean_name,
                    chunk_index=chunk_idx,
                    text=chunk_str,
                    page=page_num
                ))
                chunk_idx += 1

        if not new_chunks:
            return False, "Document contains no extractable text chunks.", 0, 0

        self.chunks.extend(new_chunks)
        doc_item = DocumentItem(
            id=doc_id,
            filename=clean_name,
            chunks_count=len(new_chunks),
            total_characters=total_chars,
            uploaded_at=time.time()
        )
        self.documents[doc_id] = doc_item
        self._rebuild_index()

        return True, "Ingested successfully", len(new_chunks), total_chars

    def _split_page(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w{2,}\b", text)]

    def _rebuild_index(self) -> None:
        self._vocabulary.clear()
        self._idf.clear()
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        doc_freq: Dict[str, int] = {}
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk.text))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        for t, df in doc_freq.items():
            self._idf[t] = math.log((doc_count + 1) / (df + 1)) + 1.0

    def query_relevant(self, query: str, top_k: int = 4, document_ids: Optional[List[str]] = None) -> List[Tuple[DocumentChunk, float]]:
        active_chunks = self.chunks
        if document_ids:
            filter_ids = set(document_ids)
            active_chunks = [c for c in self.chunks if c.doc_id in filter_ids]

        if not active_chunks or not self._idf:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        scores: List[Tuple[DocumentChunk, float]] = []
        for chunk in active_chunks:
            c_tokens = self._tokenize(chunk.text)
            c_tf: Dict[str, float] = {}
            for t in c_tokens:
                c_tf[t] = c_tf.get(t, 0) + 1

            dot = 0.0
            q_norm = 0.0
            c_norm = 0.0
            for t, qf in q_tf.items():
                w_q = qf * self._idf.get(t, 1.0)
                q_norm += w_q * w_q
                if t in c_tf:
                    w_c = c_tf[t] * self._idf.get(t, 1.0)
                    dot += w_q * w_c

            for t, cf in c_tf.items():
                w_c = cf * self._idf.get(t, 1.0)
                c_norm += w_c * w_c

            if q_norm > 0 and c_norm > 0:
                sim = dot / (math.sqrt(q_norm) * math.sqrt(c_norm))
                if sim > 0.01:
                    scores.append((chunk, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve(self, query: str, document_ids: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        High-level retrieval returning (sources, sufficient, evidence)
        compatible with legacy and modern RAG queries.
        """
        relevant = self.query_relevant(query, top_k=4, document_ids=document_ids)
        if not relevant:
            return [], False, ""

        sources: List[Dict[str, Any]] = []
        evidence_snippets: List[str] = []
        q_tokens = set(self._tokenize(query))

        for chunk, score in relevant:
            page_label = f"p. {chunk.page}" if chunk.page else "TXT"
            citation = f"{chunk.filename} ({page_label})"
            sources.append({
                "name": chunk.filename,
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "citation": citation,
                "score": round(score, 3)
            })
            evidence_snippets.append(f"[{citation}]\n{chunk.text}")

        # Determine sufficiency based on lexical overlap and similarity
        combined_text = " ".join(c.text for c, _ in relevant)
        combined_tokens = set(self._tokenize(combined_text))
        overlap_count = len(q_tokens & combined_tokens)
        sufficient = bool(relevant and overlap_count >= 1 and relevant[0][1] >= 0.15)

        return sources, sufficient, "\n\n".join(evidence_snippets)


# Global singleton instance
rag_store = DocumentKnowledgeStore()
