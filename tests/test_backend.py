import json
from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.main import app
from backend.config import settings
from backend.services.command_registry import validate_command, get_tool_definitions
from backend.services.rag_service import rag_store
from backend.services.llm_service import provider_rate_limit_message
from backend.services import llm_service

client = TestClient(app)


# --------------------------------------------------------------------------
# System & Health Endpoints
# --------------------------------------------------------------------------
def test_health_reports_status_and_version():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "provider" in data
    assert "provider_configured" in data


def test_readiness_probe():
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "ultimaitron-core"


def test_commands_list():
    response = client.get("/api/commands")
    assert response.status_code == 200
    data = response.json()
    assert "commands" in data
    cmd_names = [c["function"]["name"] for c in data["commands"]]
    assert "change_orb_mode" in cmd_names
    assert "fire_pulse" in cmd_names
    assert "trigger_burst" in cmd_names
    assert "set_orb_scale" in cmd_names


# --------------------------------------------------------------------------
# Security Headers & CORS
# --------------------------------------------------------------------------
def test_security_headers_present():
    response = client.get("/api/health")
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


# --------------------------------------------------------------------------
# Command Validation & Sanitization (Injection Prevention)
# --------------------------------------------------------------------------
def test_command_validation_safety():
    # Permitted valid mode
    valid, _, args = validate_command("change_orb_mode", {"mode": "REACTOR"})
    assert valid is True
    assert args["mode"] == "REACTOR"

    # Out-of-bounds numeric sanitization clamping
    valid, _, args = validate_command("set_orb_scale", {"scale": 99.0})
    assert valid is True
    assert args["scale"] == 1.85  # clamped to max

    # Invalid mode rejection
    valid, err, _ = validate_command("change_orb_mode", {"mode": "DESTROY_ALL"})
    assert valid is False
    assert "Invalid mode" in err

    # Arbitrary shell execution rejection
    valid, err, _ = validate_command("system_exec", {"cmd": "rm -rf /"})
    assert valid is False
    assert "not permitted" in err


# --------------------------------------------------------------------------
# Chat Endpoints & Error Handling
# --------------------------------------------------------------------------
def test_chat_rejects_blank_message():
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422

    response_stream = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": " "}]})
    assert response_stream.status_code == 422


def test_chat_explains_missing_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    response = client.post("/api/chat", json={"message": "Hello", "provider": "anthropic"})
    assert response.status_code == 503
    assert "Anthropic API key is not configured" in response.json()["detail"]


def test_provider_credit_exhaustion_is_not_misreported_as_request_rate():
    message = provider_rate_limit_message({"error": {"code": "credit_balance_exhausted"}})
    assert "run out of available credits" in message
    assert "rate limiting" not in message


def test_provider_rate_limit_has_retry_guidance():
    message = provider_rate_limit_message({"error": {"code": "rate_limit_exceeded"}})
    assert "Wait briefly" in message


def test_chat_sse_stream_success_and_offline_mock_label():
    payload = {
        "message": "Switch to reactor mode and fire a pulse",
        "history": []
    }
    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        full_text = ""
        for line in response.iter_lines():
            if line.startswith("data: "):
                raw_data = line[6:]
                events.append(raw_data)
                try:
                    ev = json.loads(raw_data)
                    if isinstance(ev, dict) and "token" in ev:
                        full_text += ev["token"]
                    elif isinstance(ev, dict) and ev.get("type") == "token":
                        pass
                except Exception:
                    pass
        assert len(events) > 0
        assert "[OFFLINE SIMULATION · MOCK AI CORE]" in full_text
        assert "Reactor Core" in full_text


# --------------------------------------------------------------------------
# Document Ingestion, RAG, & Security Limits
# --------------------------------------------------------------------------
def test_txt_upload_lists_retrieves_and_removes_document():
    rag_store.clear()
    upload = client.post(
        "/api/documents",
        files={"file": ("orb-facts.txt", b"The holographic reactor uses blue plasma containment.", "text/plain")},
    )
    assert upload.status_code == 201
    doc_res = upload.json()
    assert doc_res["status"] == "ok"
    assert doc_res["chunks_count"] >= 1

    # Check list
    listed = client.get("/api/documents").json()["documents"]
    assert any(item["name"] == "orb-facts.txt" for item in listed)

    # Retrieval
    sources, sufficient, evidence = rag_store.retrieve("What color is the plasma containment?", [doc_res["id"]])
    assert sufficient is True
    assert "orb-facts.txt" in sources[0]["citation"]
    assert "blue plasma" in evidence

    # Delete
    del_res = client.delete(f"/api/documents/{doc_res['id']}")
    assert del_res.status_code == 204


def test_document_upload_rejects_unsupported_type_and_invalid_pdf():
    unsupported = client.post("/api/documents", files={"file": ("notes.docx", b"text", "application/octet-stream")})
    assert unsupported.status_code == 400

    invalid_pdf = client.post("/api/documents", files={"file": ("broken.pdf", b"%PDF-not-a-real-pdf", "application/pdf")})
    assert invalid_pdf.status_code == 400


def test_pdf_upload_extracts_text_with_page_citation():
    rag_store.clear()
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
    doc_res = upload.json()

    sources, sufficient, evidence = rag_store.retrieve("What color light does the reactor use?", [doc_res["id"]])
    assert sufficient is True
    assert sources[0]["citation"] == "reactor.pdf (p. 1)"
    assert "violet light" in evidence
    rag_store.remove_document(doc_res["id"])


def test_clear_documents_removes_all_documents():
    client.post("/api/documents", files={"file": ("one.txt", b"first uploaded passage", "text/plain")})
    client.post("/api/documents", files={"file": ("two.txt", b"second uploaded passage", "text/plain")})
    response = client.delete("/api/documents")
    assert response.status_code == 200
    assert response.json()["removed"] >= 2
    assert len(rag_store.documents) == 0


def test_chat_with_expired_document_id_returns_410():
    rag_store.clear()
    response = client.post("/api/chat", json={
        "message": "Query regarding missing file",
        "document_ids": ["non_existent_doc_123"]
    })
    assert response.status_code == 410
