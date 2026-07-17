import logging
import pickle
import re
from pathlib import Path

import faiss
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger("ai_service")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = BASE_DIR / "vector.index"
DOCS_PATH = BASE_DIR / "docs.pkl"


class AIService:
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents: list[str] = []
        self.metadata: list[str] = []
        self._load_retrieval_assets()

    def _load_retrieval_assets(self) -> None:
        if INDEX_PATH.exists() and DOCS_PATH.exists():
            try:
                self.index = faiss.read_index(str(INDEX_PATH))
                with DOCS_PATH.open("rb") as handle:
                    documents, metadata = pickle.load(handle)
                self.documents = documents
                self.metadata = metadata
                if self.documents:
                    logger.info("Loaded %s retrieval chunks from index files", len(self.documents))
                    return
            except Exception as exc:
                logger.warning("Failed to load saved index, rebuilding from data files: %s", exc)

        self._build_from_data_files()

    def _build_from_data_files(self) -> None:
        if not DATA_DIR.exists():
            logger.warning("Data directory not found; continuing without retrieval context.")
            return

        text_chunks: list[str] = []
        metadata: list[str] = []

        for file_path in sorted(DATA_DIR.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8")
            cleaned = "\n".join(line.rstrip() for line in content.splitlines() if line.strip())
            if cleaned:
                text_chunks.append(cleaned)
                metadata.append(file_path.name)

        if not text_chunks:
            logger.warning("No text chunks found in data directory.")
            return

        self.documents = text_chunks
        self.metadata = metadata
        embeddings = self.embedding_model.encode(text_chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        self.index = index

        with DOCS_PATH.open("wb") as handle:
            pickle.dump((self.documents, self.metadata), handle)
        faiss.write_index(self.index, str(INDEX_PATH))
        logger.info("Built retrieval index from %s document chunks", len(self.documents))

    def _clean_chunk(self, chunk: str) -> str:
        if "]\n" in chunk:
            _, content = chunk.split("]\n", 1)
        else:
            content = chunk
        return content.strip()

    def _read_data_file(self, filename: str) -> str:
        path = DATA_DIR / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _extract_projects(self, text: str) -> list[tuple[str, str]]:
        projects: list[tuple[str, str]] = []
        sections = re.split(r"\n(?=Project:)", text)
        for section in sections:
            if not section.startswith("Project:"):
                continue
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            title = lines[0].replace("Project:", "", 1).strip()
            summary = ""
            for line in lines[1:]:
                if line.startswith("Description:"):
                    summary = line.replace("Description:", "", 1).strip()
                    break
                if line.startswith("Key Achievements:"):
                    summary = line.replace("Key Achievements:", "", 1).strip()
                    break
            if title:
                projects.append((title, summary or "Portfolio project"))
        return projects

    def _build_portfolio_response(self, message: str, context_chunks: list[str]) -> str:
        normalized = message.lower()
        context_text = "\n\n".join(self._clean_chunk(chunk) for chunk in context_chunks)

        if "project" in normalized:
            projects_text = self._read_data_file("projects.txt")
            projects = self._extract_projects(projects_text)
            if projects:
                lines = ["Here are some of the projects I’ve worked on:"]
                for title, summary in projects[:5]:
                    lines.append(f"- {title}: {summary}")
                return "\n".join(lines)

        if "skill" in normalized or "technology" in normalized or "tech" in normalized:
            skills_text = self._read_data_file("skills.txt")
            if skills_text:
                return skills_text[:1000]
            for chunk in context_chunks:
                cleaned = self._clean_chunk(chunk)
                if "Skills" in cleaned or "Technologies" in cleaned or "Interests" in cleaned:
                    return cleaned[:600]

        if "experience" in normalized or "work" in normalized or "career" in normalized:
            experience_text = self._read_data_file("experience.txt")
            if experience_text:
                return experience_text[:1000]
            for chunk in context_chunks:
                cleaned = self._clean_chunk(chunk)
                if "Professional Summary" in cleaned or "Career Focus" in cleaned or "Internship" in cleaned:
                    return cleaned[:800]

        if "education" in normalized:
            education_text = self._read_data_file("education.txt")
            if education_text:
                return education_text[:1000]
            for chunk in context_chunks:
                cleaned = self._clean_chunk(chunk)
                if "University" in cleaned or "Education" in cleaned:
                    return cleaned[:800]

        if "contact" in normalized or "reach" in normalized:
            contact_text = self._read_data_file("contact.txt")
            if contact_text:
                return contact_text[:1000]
            for chunk in context_chunks:
                cleaned = self._clean_chunk(chunk)
                if "Contact" in cleaned or "Email" in cleaned or "Phone" in cleaned:
                    return cleaned[:800]

        if "about" in normalized or "who" in normalized:
            about_text = self._read_data_file("about.txt")
            if about_text:
                return about_text[:1000]
            for chunk in context_chunks:
                cleaned = self._clean_chunk(chunk)
                if "Profile" in cleaned or "Professional Summary" in cleaned:
                    return cleaned[:800]

        return (
            "Here’s a concise summary from my portfolio:\n\n"
            f"{context_text[:1200]}"
        )

    def _build_prompt(self, message: str, context_chunks: list[str]) -> str:
        context_text = "\n\n".join(self._clean_chunk(chunk) for chunk in context_chunks) if context_chunks else "No specific context was found in the available portfolio data."
        return (
            "You are representing Vinod Bavage in a professional portfolio assistant role. "
            "Answer as Vinod in a clear, concise, and natural way. "
            "Use the provided context as the only source of truth. "
            "Do not answer as a generic chatbot or mention that you are a language model. "
            "If the context does not contain enough information, say you do not know rather than guessing. "
            "Write a polished response with a short intro, then use bullets when helpful. "
            "Keep the answer professional, friendly, and easy to read. "
            "Always ground the answer in the context and never invent details.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question:\n{message}\n\n"
            "Reply in a friendly, professional tone and speak in first person."
        )

    def _retrieve_context(self, message: str, top_k: int = 6) -> list[str]:
        if self.index is None or not self.documents:
            return []

        try:
            embedding = self.embedding_model.encode([message])
            _, indices = self.index.search(embedding, top_k)
            retrieved: list[str] = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    source = self.metadata[idx]
                    retrieved.append(f"[Source: {source}]\n{self.documents[idx]}")
            return retrieved
        except Exception as exc:
            logger.exception("Vector search failed")
            return []

    async def generate_response(self, message: str) -> str:
        if not self.client:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        context_chunks = self._retrieve_context(message, top_k=6)
        if context_chunks:
            return self._build_portfolio_response(message, context_chunks)

        prompt = self._build_prompt(message, context_chunks)

        system_prompt = (
            "You are a polished portfolio assistant for Vinod Bavage. "
            "Give concise, professional, and friendly answers. "
            "Stay on topic, use the provided context, and answer in first person as Vinod. "
            "Prefer short, helpful responses with clear formatting."
        )

        try:
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=350,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "I’m not sure how to answer that."
        except Exception as exc:
            logger.exception("Groq request failed")
            raise RuntimeError("AI service is temporarily unavailable.") from exc


ai_service = AIService()
