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
        description="GOOGLE API KEY."
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

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
