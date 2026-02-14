"""
Centralized Configuration for the LégiRoute RAG system.

All runtime constants and paths shared across modules are defined here.
This avoids scattering magic values across the codebase.

Usage:
    from src.config import settings
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    # --- API ---
    GOOGLE_API_KEY: str = Field(
        default="",
        description="Google AI Studio API key."
    )

    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    @property
    def RAW_DATA_DIR(self) -> Path:
        return (
            self.PROJECT_ROOT / "data" / "raw" / "LEGI" / "TEXT"
            / "00" / "00" / "06" / "07" / "42"
            / "LEGITEXT000006074228" / "article"
        )

    @property
    def PROCESSED_FILE(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed" / "code_route_articles.json"

    @property
    def CHROMA_DB_PATH(self) -> Path:
        return self.PROJECT_ROOT / "data" / "chroma_db"

    # --- Vector Database ---
    COLLECTION_NAME: str = "traffic_law_v1"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- Indexing Pipeline ---
    BATCH_SIZE: int = 5
    SLEEP_BETWEEN_BATCHES: int = 5
    MAX_RETRIES: int = 20
    RETRY_MIN_WAIT: int = 10
    RETRY_MAX_WAIT: int = 120

    # --- Retrieval ---
    DEFAULT_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 1.1

    # --- Generation ---
    GENERATION_MODEL: str = "models/gemini-2.5-flash"
    GENERATION_TEMPERATURE: float = 0.0
    GENERATION_MAX_TOKENS: int = 1000

    # --- Chitchat Detection ---
    CHITCHAT_KEYWORDS: list[str] = [
        "bonjour", "salut", "hello", "qui es-tu",
        "tu es qui", "ton nom", "merci"
    ]
    MAX_CHITCHAT_LENGTH: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
