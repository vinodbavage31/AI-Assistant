import logging
from typing import Any

from groq import AsyncGroq

from config.settings import settings

logger = logging.getLogger("ai_service")


class AIService:
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    async def generate_response(self, message: str) -> str:
        if not self.client:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        system_prompt = (
            "You are a professional AI assistant. "
            "Answer clearly, concisely, and helpfully. "
            "Stay focused on the user's question. "
            "If you are unsure, say you do not know instead of guessing."
        )

        try:
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "I’m not sure how to answer that."
        except Exception as exc:
            logger.exception("Groq request failed")
            raise RuntimeError("AI service is temporarily unavailable.") from exc


ai_service = AIService()
