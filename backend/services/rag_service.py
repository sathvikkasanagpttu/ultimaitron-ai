import os
import io
import math
import re
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
from backend.config import settings
from backend.schemas import DocumentChunk, DocumentItem

class DocumentKnowledgeStore:
    def __init__(self):
        self.documents: Dict[str, DocumentItem] = {}
        self.chunks: List[DocumentChunk] = []
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self._vocabulary.clear()
        self._idf.clear()

    def get_document_list(self) -> List[DocumentItem]:
        return list(self.documents.values())

    def remove_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            self._rebuild_index()
            return True
        return False

    def ingest_document(self, filename: str, content_bytes: bytes) -> Tuple[bool, str, int, int]:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".txt", ".pdf", ".md", ".json"]:
            return False, f"Unsupported file extension '{ext}'. Only .txt, .pdf, .md, .json are supported.", 0, 0

        max_bytes = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
        if len(content_bytes) > max_bytes:
            return False, f"File exceeds maximum allowed size of {settings.MAX_DOCUMENT_SIZE_MB}MB.", 0, 0

        raw_text = ""
        try:
            if ext == ".pdf":
                reader = PdfReader(io.BytesIO(content_bytes))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
            else:
                raw_text = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return False, f"Error decoding document: {str(e)}", 0, 0

        raw_text = raw_text.strip()
        if not raw_text:
            return False, "Document contains no extractable text.", 0, 0

        doc_id = str(uuid.uuid4())[:8]
        chunks_text = self._split_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        
        for idx, chunk_str in enumerate(chunks_text):
            self.chunks.append(DocumentChunk(
                doc_id=doc_id,
                filename=filename,
                chunk_index=idx,
                text=chunk_str
            ))

        doc_item = DocumentItem(
            id=doc_id,
            filename=filename,
            chunks_count=len(chunks_text),
            total_characters=len(raw_text),
            uploaded_at=time.time()
        )
        self.documents[doc_id] = doc_item
        self._rebuild_index()

        return True, "Ingested successfully", len(chunks_text), len(raw_text)

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
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

    def query_relevant(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or not self._idf:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        
        scores: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            c_tokens = self._tokenize(chunk.text)
            c_tf: Dict[str, float] = {}
            for t in c_tokens:
                c_tf[t] = c_tf.get(t, 0) + 1

            dot = 0.0
            q_norm = 0.0
            c_norm = 0.0

            for t, count in q_tf.items():
                idf = self._idf.get(t, 1.0)
                q_val = count * idf
                q_norm += q_val * q_val
                if t in c_tf:
                    c_val = c_tf[t] * idf
                    dot += q_val * c_val

            for t, count in c_tf.items():
                idf = self._idf.get(t, 1.0)
                c_val = count * idf
                c_norm += c_val * c_val

            if q_norm > 0 and c_norm > 0:
                score = dot / (math.sqrt(q_norm) * math.sqrt(c_norm))
            else:
                score = 0.0

            if score > 0.05:
                scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

rag_store = DocumentKnowledgeStore()
