from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.app import main as main_module
from backend.app.main import app, documents, provider_rate_limit_message, settings

client = TestClient(app)


def test_health_reports_provider_configuration(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["provider_configured"] is False


def test_chat_rejects_blank_message():
    response = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": " "}]})
    assert response.status_code == 422


def test_chat_explains_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    response = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 503


def test_provider_credit_exhaustion_is_not_misreported_as_request_rate():
    message = provider_rate_limit_message({"error": {"code": "credit_balance_exhausted"}})
    assert "run out of available credits" in message
    assert "rate limiting" not in message


def test_provider_rate_limit_has_retry_guidance():
    message = provider_rate_limit_message({"error": {"code": "rate_limit_exceeded"}})
    assert "Wait briefly" in message


def test_txt_upload_lists_retrieves_and_removes_document():
    upload = client.post(
        "/api/documents",
        files={"file": ("orb-facts.txt", b"The holographic reactor uses blue plasma containment.", "text/plain")},
    )
    assert upload.status_code == 201
    document = upload.json()
    assert document["status"] == "ready"
    listed = client.get("/api/documents").json()["documents"]
    assert any(item["id"] == document["id"] for item in listed)
    sources, sufficient, evidence = documents.retrieve("What color is the plasma containment?", [document["id"]])
    assert sufficient is True
    assert sources[0]["citation"] == "orb-facts.txt (TXT)"
    assert "blue plasma" in evidence
    assert client.delete(f"/api/documents/{document['id']}").status_code == 204


def test_document_upload_rejects_unsupported_type_and_invalid_pdf():
    unsupported = client.post("/api/documents", files={"file": ("notes.docx", b"text", "application/octet-stream")})
    assert unsupported.status_code == 400
    invalid_pdf = client.post("/api/documents", files={"file": ("broken.pdf", b"%PDF-not-a-real-pdf", "application/pdf")})
    assert invalid_pdf.status_code == 400


def test_clear_documents_removes_all_documents():
    first = client.post("/api/documents", files={"file": ("one.txt", b"first uploaded passage", "text/plain")}).json()
    second = client.post("/api/documents", files={"file": ("two.txt", b"second uploaded passage", "text/plain")}).json()
    response = client.delete("/api/documents")
    assert response.status_code == 200
    assert response.json()["removed"] >= 2
    assert not documents.contains([first["id"], second["id"]])


def test_pdf_upload_extracts_text_with_page_citation():
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font_ref = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 250 Td (The reactor uses violet light.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    pdf = BytesIO()
    writer.write(pdf)

    upload = client.post(
        "/api/documents",
        files={"file": ("reactor.pdf", pdf.getvalue(), "application/pdf")},
    )
    assert upload.status_code == 201
    document = upload.json()
    sources, sufficient, evidence = documents.retrieve("What color light does the reactor use?", [document["id"]])
    assert sufficient is True
    assert sources[0]["citation"] == "reactor.pdf (p. 1)"
    assert "violet light" in evidence
    documents.remove(document["id"])


def test_streamed_provider_429_returns_credit_guidance(monkeypatch):
    class ProviderResponse:
        status_code = 429

        async def aread(self):
            return b'{"error":{"code":"credit_balance_exhausted"}}'

    class StreamContext:
        async def __aenter__(self):
            return ProviderResponse()

        async def __aexit__(self, *args):
            return None

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, *args, **kwargs):
            return StreamContext()

    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(main_module.httpx, "AsyncClient", AsyncClient)
    response = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert "run out of available credits" in response.text
    assert "event: error" in response.text


def test_transient_provider_failure_is_retried_before_streaming(monkeypatch):
    class ProviderResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"ready"}}]}'
            yield "data: [DONE]"

    class StreamContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return None

    class AsyncClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, *args, **kwargs):
            self.calls += 1
            status = 503 if self.calls == 1 else 200
            return StreamContext(ProviderResponse(status))

    fake_client = AsyncClient()

    class ClientFactory(AsyncClient):
        def __new__(cls, *args, **kwargs):
            return fake_client

    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(main_module.httpx, "AsyncClient", ClientFactory)
    response = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert fake_client.calls == 2
    assert '"token": "ready"' in response.text
    assert "event: done" in response.text
