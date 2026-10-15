from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

PERMITTED_COMMANDS = {
    "change_orb_mode": {
        "description": "Switch the 3D holographic orb configuration mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["NEURAL", "REACTOR", "DATA", "ENERGY", "SCANNER"],
                    "description": "Target configuration mode."
                }
            },
            "required": ["mode"]
        }
    },
    "fire_pulse": {
        "description": "Trigger an energy wavefront pulse propagating through the neural core.",
        "parameters": {
            "type": "object",
            "properties": {
                "intensity": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 2.0,
                    "description": "Pulse wavefront energy intensity."
                }
            }
        }
    },
    "trigger_burst": {
        "description": "Detonate a power surge shockwave with particle burst.",
        "parameters": {
            "type": "object",
            "properties": {
                "warm": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "0 = cyan shockwave, 1 = golden energy blast."
                }
            }
        }
    },
    "set_orb_scale": {
        "description": "Adjust the holographic orb physical scale within safe visual bounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "minimum": 0.62,
                    "maximum": 1.85,
                    "description": "Scale factor."
                }
            },
            "required": ["scale"]
        }
    },
    "set_warmth": {
        "description": "Shift the neural core energy spectrum between cyan and deep gold.",
        "parameters": {
            "type": "object",
            "properties": {
                "warmth": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "0 = cool cyan, 1 = hot gold."
                }
            },
            "required": ["warmth"]
        }
    },
    "open_panel": {
        "description": "Open or toggle a specific HUD panel.",
        "parameters": {
            "type": "object",
            "properties": {
                "panel": {
                    "type": "string",
                    "enum": ["chat", "diagnostics", "knowledge"],
                    "description": "The panel name to toggle."
                }
            },
            "required": ["panel"]
        }
    },
    "reset_system": {
        "description": "Reset orb orientation, scale, and mode to baseline default.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

def validate_command(name: str, arguments: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
    """
    Validates that a command is in the explicit allowlist and its arguments
    conform to safe bounds. Never executes arbitrary code.
    """
    if name not in PERMITTED_COMMANDS:
        return False, f"Command '{name}' is not permitted.", {}

    sanitized = {}
    if name == "change_orb_mode":
        mode = str(arguments.get("mode", "")).upper()
        if mode not in ["NEURAL", "REACTOR", "DATA", "ENERGY", "SCANNER"]:
            return False, f"Invalid mode '{mode}'. Allowed: NEURAL, REACTOR, DATA, ENERGY, SCANNER.", {}
        sanitized["mode"] = mode

    elif name == "fire_pulse":
        intensity = float(arguments.get("intensity", 1.0))
        sanitized["intensity"] = max(0.1, min(2.0, intensity))

    elif name == "trigger_burst":
        warm = float(arguments.get("warm", 0.0))
        sanitized["warm"] = max(0.0, min(1.0, warm))

    elif name == "set_orb_scale":
        scale = float(arguments.get("scale", 1.0))
        sanitized["scale"] = max(0.62, min(1.85, scale))

    elif name == "set_warmth":
        warmth = float(arguments.get("warmth", 0.0))
        sanitized["warmth"] = max(0.0, min(1.0, warmth))

    elif name == "open_panel":
        panel = str(arguments.get("panel", "")).lower()
        if panel not in ["chat", "diagnostics", "knowledge"]:
            return False, f"Invalid panel '{panel}'. Allowed: chat, diagnostics, knowledge.", {}
        sanitized["panel"] = panel

    elif name == "reset_system":
        sanitized = {}

    return True, "Validated", sanitized

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Returns OpenAI/Anthropic compatible tool schema specifications."""
    schemas = []
    for name, spec in PERMITTED_COMMANDS.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"]
            }
        })
    return schemas
