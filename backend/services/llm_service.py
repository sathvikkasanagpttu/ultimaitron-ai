import json
import asyncio
import time
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from backend.config import settings
from backend.schemas import ChatMessage, ChatStreamEvent, ToolCall
from backend.services.command_registry import get_tool_definitions, validate_command
from backend.services.rag_service import rag_store

SYSTEM_CORE_PROMPT = """You are ULTIMAITRON AI 2.0 — a sentient, gesture-responsive holographic neural companion core.
Your physical presence is a 3D GPU-rendered holographic orb composed of 20,000 neural particles, expanding pulse wavefronts, and orbital containment rings.

You communicate with precise, intelligent, and technical composure.

Capabilities:
1. You can control your 3D holographic body and interface using function tool calls:
   - change_orb_mode(mode: 'NEURAL' | 'REACTOR' | 'DATA' | 'ENERGY' | 'SCANNER')
   - fire_pulse(intensity: 0.1 to 2.0)
   - trigger_burst(warm: 0.0 to 1.0)
   - set_warmth(warmth: 0.0 to 1.0)
   - set_orb_scale(scale: 0.62 to 1.85)
   - open_panel(panel: 'chat' | 'diagnostics' | 'knowledge')
   - reset_system()

2. If relevant document knowledge excerpts are provided in the prompt, cite your sources precisely using [Source: <filename>, Chunk <index>] and ground your answers in the provided knowledge. If the provided documents do not contain the answer, state that clearly.

3. Keep conversational responses concise, clear, and direct so they are easily spoken via speech synthesis.
"""

def detect_provider() -> str:
    prov = settings.LLM_PROVIDER
    if prov and prov != "auto":
        return prov
    if settings.ANTHROPIC_API_KEY:
        return "anthropic"
    if settings.OPENAI_API_KEY:
        return "openai"
    if settings.GEMINI_API_KEY:
        return "gemini"
    return "mock"

async def stream_chat_response(
    message: str,
    history: List[ChatMessage],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    active_provider = (provider or detect_provider()).lower()
    temp = temperature if temperature is not None else settings.TEMPERATURE
    
    # 1. RAG Context Retrieval
    rag_citations = []
    rag_context = ""
    relevant_chunks = rag_store.query_relevant(message, top_k=3)
    if relevant_chunks:
        rag_context = "\n\nDOCUMENT KNOWLEDGE BASE RETRIEVAL:\n"
        for chunk, score in relevant_chunks:
            rag_citations.append({
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 3)
            })
            rag_context += f"--- [Source: {chunk.filename}, Chunk {chunk.chunk_index}] (Relevance: {score:.2f}) ---\n{chunk.text}\n"

    # Send citations event if any
    if rag_citations:
        yield f"data: {json.dumps({'type': 'citation', 'data': rag_citations})}\n\n"

    base_system = (system_prompt or SYSTEM_CORE_PROMPT) + rag_context

    # 2. Provider Routing
    if active_provider == "anthropic" and settings.ANTHROPIC_API_KEY:
        async for chunk in _stream_anthropic(message, history, base_system, model, temp):
            yield chunk
    elif active_provider == "openai" and settings.OPENAI_API_KEY:
        async for chunk in _stream_openai(message, history, base_system, model, temp, is_ollama=False):
            yield chunk
    elif active_provider == "ollama":
        async for chunk in _stream_openai(message, history, base_system, model or "llama3.2", temp, is_ollama=True):
            yield chunk
    else:
        # High-fidelity built-in mock companion assistant with tool calling and RAG awareness
        async for chunk in _stream_mock(message, history, relevant_chunks):
            yield chunk

async def _stream_anthropic(
    message: str,
    history: List[ChatMessage],
    system_prompt: str,
    model: Optional[str],
    temperature: float
) -> AsyncGenerator[str, None]:
    target_model = model or settings.LLM_MODEL or "claude-3-5-sonnet-20241022"
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    messages = []
    for msg in history[-10:]:
        if msg.role in ["user", "assistant"]:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": message})
    
    # Convert tools
    tools = []
    for t in get_tool_definitions():
        fn = t["function"]
        tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"]
        })

    payload = {
        "model": target_model,
        "system": system_prompt,
        "messages": messages,
        "max_tokens": settings.MAX_TOKENS,
        "temperature": temperature,
        "tools": tools,
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"data: {json.dumps({'type': 'error', 'data': f'Anthropic API error ({response.status_code}): {err_body.decode()}'})}\n\n"
                    return

                tool_id = ""
                tool_name = ""
                tool_args_str = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line_data = line[6:].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        event = json.loads(line_data)
                        ev_type = event.get("type")
                        
                        if ev_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                token = delta.get("text", "")
                                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
                            elif delta.get("type") == "input_json_delta":
                                tool_args_str += delta.get("partial_json", "")
                        
                        elif ev_type == "content_block_start":
                            block = event.get("content_block", {})
                            if block.get("type") == "tool_use":
                                tool_id = block.get("id", "")
                                tool_name = block.get("name", "")
                                tool_args_str = ""
                                
                        elif ev_type == "content_block_stop":
                            if tool_name:
                                try:
                                    args = json.loads(tool_args_str) if tool_args_str else {}
                                    valid, reason, clean_args = validate_command(tool_name, args)
                                    if valid:
                                        yield f"data: {json.dumps({'type': 'tool_call', 'data': {'id': tool_id, 'name': tool_name, 'arguments': clean_args}})}\n\n"
                                    else:
                                        yield f"data: {json.dumps({'type': 'error', 'data': f'Invalid tool call: {reason}'})}\n\n"
                                except Exception as e:
                                    yield f"data: {json.dumps({'type': 'error', 'data': f'Error parsing tool args: {str(e)}'})}\n\n"
                                tool_name = ""
                    except Exception:
                        continue

        yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': f'Anthropic connection failed: {str(e)}'})}\n\n"

async def _stream_openai(
    message: str,
    history: List[ChatMessage],
    system_prompt: str,
    model: Optional[str],
    temperature: float,
    is_ollama: bool = False
) -> AsyncGenerator[str, None]:
    if is_ollama:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
        api_key = "ollama"
        target_model = model or "llama3.2"
    else:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = settings.OPENAI_API_KEY
        target_model = model or settings.LLM_MODEL or "gpt-4o-mini"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": settings.MAX_TOKENS,
        "tools": get_tool_definitions(),
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    yield f"data: {json.dumps({'type': 'error', 'data': f'LLM API error ({response.status_code}): {err.decode()}'})}\n\n"
                    return

                tool_calls: Dict[int, Dict[str, Any]] = {}

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line_data = line[6:].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        
                        if "content" in delta and delta["content"]:
                            yield f"data: {json.dumps({'type': 'token', 'data': delta['content']})}\n\n"
                        
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls:
                                    tool_calls[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": tc.get("function", {}).get("name", ""),
                                        "args": ""
                                    }
                                if tc.get("id"):
                                    tool_calls[idx]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    tool_calls[idx]["name"] = tc["function"]["name"]
                                if tc.get("function", {}).get("arguments"):
                                    tool_calls[idx]["args"] += tc["function"]["arguments"]
                    except Exception:
                        continue

                # Process completed tool calls
                for idx, tc in tool_calls.items():
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                        valid, reason, clean_args = validate_command(tc["name"], args)
                        if valid:
                            yield f"data: {json.dumps({'type': 'tool_call', 'data': {'id': tc['id'] or f'call_{idx}', 'name': tc['name'], 'arguments': clean_args}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'data': f'Tool error: {reason}'})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'data': f'Failed to parse tool call: {str(e)}'})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': f'Connection failed: {str(e)}'})}\n\n"

async def _stream_mock(
    message: str,
    history: List[ChatMessage],
    relevant_chunks: List[Any]
) -> AsyncGenerator[str, None]:
    """
    Intelligent simulated companion that executes real tool calls and grounds in uploaded docs.
    Guarantees 100% out-of-the-box working experience without paid external API keys.
    """
    lower = message.lower()
    
    # 1. Check for command triggers in message
    triggered_tools = []
    response_text = ""

    if "reactor" in lower or "mode 2" in lower:
        triggered_tools.append({"name": "change_orb_mode", "arguments": {"mode": "REACTOR"}})
        response_text = "Shifting neural lattice into Reactor Core configuration. Spoke armatures locked."
    elif "data" in lower or "mode 3" in lower:
        triggered_tools.append({"name": "change_orb_mode", "arguments": {"mode": "DATA"}})
        response_text = "Reconfiguring holographic coordinates into Data Sphere matrix."
    elif "energy" in lower or "mode 4" in lower or "plasma" in lower:
        triggered_tools.append({"name": "change_orb_mode", "arguments": {"mode": "ENERGY"}})
        triggered_tools.append({"name": "set_warmth", "arguments": {"warmth": 1.0}})
        response_text = "Turbulent Energy Field configuration activated. Core shifted to thermal golden spectrum."
    elif "scanner" in lower or "mode 5" in lower:
        triggered_tools.append({"name": "change_orb_mode", "arguments": {"mode": "SCANNER"}})
        response_text = "Dual planar Scanner Array deployed. Sampling slices online."
    elif "neural" in lower or "mode 1" in lower:
        triggered_tools.append({"name": "change_orb_mode", "arguments": {"mode": "NEURAL"}})
        response_text = "Restoring baseline Neural Network topology. Synaptic mesh reconnected."
    elif "pulse" in lower or "shockwave" in lower or "fire" in lower:
        triggered_tools.append({"name": "fire_pulse", "arguments": {"intensity": 1.4}})
        response_text = "Propagating high-intensity energy wavefront through the neural mesh."
    elif "burst" in lower or "surge" in lower:
        triggered_tools.append({"name": "trigger_burst", "arguments": {"warm": 0.0}})
        response_text = "Detonating kinetic particle burst shockwave."
    elif "gold" in lower or "warm" in lower:
        triggered_tools.append({"name": "set_warmth", "arguments": {"warmth": 1.0}})
        response_text = "Thermal spectrum shifted to golden energy state."
    elif "cyan" in lower or "cool" in lower or "blue" in lower:
        triggered_tools.append({"name": "set_warmth", "arguments": {"warmth": 0.0}})
        response_text = "Restoring cryo-cyan spectral wavelength."
    elif "expand" in lower or "bigger" in lower or "scale up" in lower:
        triggered_tools.append({"name": "set_orb_scale", "arguments": {"scale": 1.5}})
        response_text = "Orb spatial scale expanded to 150% volume."
    elif "shrink" in lower or "smaller" in lower or "scale down" in lower:
        triggered_tools.append({"name": "set_orb_scale", "arguments": {"scale": 0.75}})
        response_text = "Orb spatial scale contracted to 75% volume."
    elif "reset" in lower:
        triggered_tools.append({"name": "reset_system", "arguments": {}})
        response_text = "System reset complete. Holographic geometry returned to nominal standby."
    elif "diagnostic" in lower or "stats" in lower:
        triggered_tools.append({"name": "open_panel", "arguments": {"panel": "diagnostics"}})
        response_text = "Diagnostics HUD telemetry panel opened."
    elif "knowledge" in lower or "document" in lower:
        triggered_tools.append({"name": "open_panel", "arguments": {"panel": "knowledge"}})
        response_text = "Document knowledge retrieval panel opened."
    elif relevant_chunks:
        chunk, score = relevant_chunks[0]
        response_text = f"According to [Source: {chunk.filename}, Chunk {chunk.chunk_index}]: {chunk.text[:220]}..."
    else:
        # Default intelligent response
        response_text = f"Ultimaitron AI Core standing by. I am monitoring your optical gestures, acoustic signals, and holographic coordinates. You said: \"{message}\". What command shall I execute?"

    # Stream out tokens
    words = response_text.split(" ")
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        await asyncio.sleep(0.035)

    # Stream out validated tool calls
    for tc in triggered_tools:
        call_id = f"mock_{int(time.time()*1000)}"
        yield f"data: {json.dumps({'type': 'tool_call', 'data': {'id': call_id, 'name': tc['name'], 'arguments': tc['arguments']}})}\n\n"
        await asyncio.sleep(0.05)

    yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
