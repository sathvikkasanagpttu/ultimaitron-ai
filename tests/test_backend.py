import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.command_registry import validate_command
from backend.services.rag_service import rag_store

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "provider" in data

def test_commands_list():
    response = client.get("/api/commands")
    assert response.status_code == 200
    data = response.json()
    assert "commands" in data
    cmd_names = [c["function"]["name"] for c in data["commands"]]
    assert "change_orb_mode" in cmd_names
    assert "fire_pulse" in cmd_names
    assert "trigger_burst" in cmd_names

def test_command_validation_safety():
    # Permitted valid
    valid, _, args = validate_command("change_orb_mode", {"mode": "REACTOR"})
    assert valid is True
    assert args["mode"] == "REACTOR"

    # Permitted with sanitization clamping
    valid, _, args = validate_command("set_orb_scale", {"scale": 99.0})
    assert valid is True
    assert args["scale"] == 1.85  # clamped to max

    # Invalid mode
    valid, err, _ = validate_command("change_orb_mode", {"mode": "DESTROY_ALL"})
    assert valid is False
    assert "Invalid mode" in err

    # Unpermitted command (rejecting arbitrary execution)
    valid, err, _ = validate_command("system_exec", {"cmd": "rm -rf /"})
    assert valid is False
    assert "not permitted" in err

def test_chat_sse_stream():
    payload = {
        "message": "Switch to reactor mode and fire a pulse",
        "history": []
    }
    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
        assert len(events) > 0

def test_rag_document_workflow():
    rag_store.clear()
    
    # Upload sample text document
    file_content = b"The Ultimaitron AI core was designed with a dual-layer holographic shader and 20,000 GPU particles."
    files = {"file": ("manual.txt", file_content, "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["chunks_count"] >= 1

    # Check list
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    # Query RAG
    results = rag_store.query_relevant("How many GPU particles does Ultimaitron have?", top_k=1)
    assert len(results) == 1
    assert "20,000" in results[0][0].text

    # Clear
    clear_res = client.post("/api/documents/clear")
    assert clear_res.status_code == 200
    assert len(rag_store.documents) == 0
