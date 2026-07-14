import socket

from src.config import (
    OLLAMA_MODEL,
    GROQ_VISION_MODEL
)


def internet_available():
    """Return True if internet is available."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


def route_model(chunks=None, override=None):
    """
    Select which model provider should handle the request.

    Priority:
    1. Manual override.
    2. If internet is available, use Groq.
    3. Otherwise, use the local Ollama model.

    If Groq later fails, answer_generator.py automatically
    falls back to Ollama.
    """

    if override == "ollama":
        return {
            "provider": "ollama",
            "model": OLLAMA_MODEL
        }

    if override == "groq":
        return {
            "provider": "groq",
            "model": GROQ_VISION_MODEL
        }

    if internet_available():
        return {
            "provider": "groq",
            "model": GROQ_VISION_MODEL
        }

    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL
    }