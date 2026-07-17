import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


class Settings:
    GROQ_API_KEY: str | None = _get_env("GROQ_API_KEY")
    GROQ_MODEL: str = _get_env("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant"
    APP_TITLE: str = _get_env("APP_TITLE", "AI Assistant API") or "AI Assistant API"
    APP_VERSION: str = _get_env("APP_VERSION", "1.0.0") or "1.0.0"


settings = Settings()
