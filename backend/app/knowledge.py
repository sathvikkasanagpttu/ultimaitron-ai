import io
import re
from dataclasses import dataclass
from uuid import uuid4

from pypdf import PdfReader

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOCUMENTS = 10
MAX_EXTRACTED_CHARS = 1_000_000
MAX_PDF_PAGES = 1000
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None


@dataclass(frozen=True)
class StoredDocument:
    id: str
    name: str
    chunks: tuple[Chunk, ...]


class DocumentStore:
    """Small process-local store using transparent lexical retrieval, no embeddings."""

    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}

    @property
    def documents(self) -> list[StoredDocument]:
        return list(self._documents.values())

    def add(self, filename: str, content: bytes) -> StoredDocument:
        if len(self._documents) >= MAX_DOCUMENTS:
            raise DocumentError("At most 10 documents can be kept in memory at once. Remove a document and retry.")
        name = _safe_filename(filename)
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in {"txt", "pdf"}:
            raise DocumentError("Only TXT and PDF documents are supported.")
        if not content:
            raise DocumentError("The uploaded file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise DocumentError("File exceeds the 10 MB upload limit.")

        if suffix == "pdf":
            if not content.startswith(b"%PDF-"):
                raise DocumentError("The file does not have a valid PDF signature.")
            try:
                reader = PdfReader(io.BytesIO(content), strict=False)
                if len(reader.pages) > MAX_PDF_PAGES:
                    raise DocumentError("PDF exceeds the 1,000 page processing limit.")
                pages = [(page.extract_text() or "", number) for number, page in enumerate(reader.pages, 1)]
            except DocumentError:
                raise
            except Exception as error:
                raise DocumentError("Could not read this PDF. Check that it is not encrypted or damaged.") from error
            indexed = _chunks_with_pages(pages)
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise DocumentError("TXT files must use UTF-8 encoding.") from error
            indexed = _chunks_with_pages([(text, None)])

        if not indexed:
            raise DocumentError("No readable text was found in this document.")
        document = StoredDocument(id=uuid4().hex, name=name, chunks=tuple(indexed))
        self._documents[document.id] = document
        return document

    def remove(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

    def contains(self, document_ids: list[str]) -> bool:
        return all(document_id in self._documents for document_id in document_ids)

    def clear(self) -> int:
        count = len(self._documents)
        self._documents.clear()
        return count

    def retrieve(self, query: str, document_ids: list[str]) -> tuple[list[dict[str, str | int | float | None]], bool, str]:
        selected = [self._documents[doc_id] for doc_id in dict.fromkeys(document_ids) if doc_id in self._documents]
        if not selected:
            return [], False, ""

        terms = set(_terms(query))
        if not terms:
            return [], False, ""
        matches: list[tuple[float, StoredDocument, Chunk]] = []
        for document in selected:
            for chunk in document.chunks:
                chunk_terms = set(_terms(chunk.text))
                overlap = terms & chunk_terms
                if overlap:
                    matches.append((len(overlap) / len(terms), document, chunk))
        matches.sort(key=lambda item: item[0], reverse=True)
        top = matches[:4]
        sufficient = bool(top and len(terms & set(_terms(" ".join(chunk.text for _, _, chunk in top)))) >= 2 and top[0][0] >= 0.2)
        sources: list[dict[str, str | int | float | None]] = []
        evidence: list[str] = []
        for score, document, chunk in top:
            page_label = f"p. {chunk.page}" if chunk.page else "TXT"
            label = f"{document.name} ({page_label})"
            sources.append({"name": document.name, "page": chunk.page, "citation": label, "score": round(score, 3)})
            evidence.append(f"[{label}]\n{chunk.text}")
        return sources, sufficient, "\n\n".join(evidence)


def _safe_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").split("/")[-1].strip()
    cleaned = re.sub(r"[^\w .()-]", "_", basename, flags=re.UNICODE).strip(" .")
    return (cleaned or "document")[:160]


def _chunks_with_pages(pages: list[tuple[str, int | None]]) -> list[Chunk]:
    output: list[Chunk] = []
    extracted_size = 0
    for text, page in pages:
        normalized = re.sub(r"\s+", " ", text).strip()
        extracted_size += len(normalized)
        if extracted_size > MAX_EXTRACTED_CHARS:
            raise DocumentError("Extracted text exceeds the 1 MB processing limit.")
        start = 0
        while start < len(normalized):
            snippet = normalized[start : start + CHUNK_SIZE].strip()
            if snippet:
                output.append(Chunk(snippet, page))
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return output


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w'-]{2,}", text.casefold())
