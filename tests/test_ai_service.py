import asyncio
import socket

from app import get_port
from main import get_available_port
from services.ai_service import AIService


def test_build_prompt_includes_context_and_user_message() -> None:
    service = AIService.__new__(AIService)
    prompt = service._build_prompt("Tell me about your projects", ["Project one", "Project two"])

    assert "Tell me about your projects" in prompt
    assert "Project one" in prompt
    assert "Project two" in prompt
    assert "Context" in prompt


def test_generate_response_uses_portfolio_fallback_without_groq_key(monkeypatch) -> None:
    service = AIService.__new__(AIService)
    service.client = None
    service.embedding_model = None
    service.index = None
    service.documents = []
    service.metadata = []

    monkeypatch.setattr(service, "_retrieve_context", lambda message, top_k=6: ["[Source: about.txt]\nI build apps."])
    monkeypatch.setattr(service, "_build_portfolio_response", lambda message, context_chunks: "Portfolio fallback response")

    response = asyncio.run(service.generate_response("Tell me about yourself"))

    assert response == "Portfolio fallback response"


def test_get_port_uses_render_port(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "8080")
    assert get_port() == 8080


def test_get_port_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    assert get_port() == 8000


def test_get_available_port_returns_free_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        occupied_port = sock.getsockname()[1]

    result = get_available_port(occupied_port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", result))
