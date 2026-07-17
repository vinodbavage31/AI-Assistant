from services.ai_service import AIService


def test_build_prompt_includes_context_and_user_message() -> None:
    service = AIService.__new__(AIService)
    prompt = service._build_prompt("Tell me about your projects", ["Project one", "Project two"])

    assert "Tell me about your projects" in prompt
    assert "Project one" in prompt
    assert "Project two" in prompt
    assert "Context" in prompt
