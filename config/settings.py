import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or None
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    APP_TITLE: str = os.getenv("APP_TITLE", "AI Assistant API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")


settings = Settings()
