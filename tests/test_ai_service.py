import asyncio

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
