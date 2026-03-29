from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Provider(str, Enum):
    GEMINI = "gemini"


PROVIDER_MODELS = {
    Provider.GEMINI: {
        "classifier": "models/gemini-2.5-flash-lite",
        "generation": "models/gemini-2.5-flash",
        "embedding": "gemini-embedding-001",
    },
}


class Settings(BaseSettings):

    GOOGLE_API_KEY: str = ""
    PROVIDER: Provider = Provider.GEMINI

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

    @property
    def CLASSIFIER_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["classifier"]

    @property
    def GENERATION_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["generation"]

    @property
    def EMBEDDING_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["embedding"]

    @property
    def COLLECTION_NAME(self) -> str:
        return "traffic_law_v1"

    # Indexing
    BATCH_SIZE: int = 5
    SLEEP_BETWEEN_BATCHES: int = 5
    MAX_RETRIES: int = 20
    RETRY_MIN_WAIT: int = 10
    RETRY_MAX_WAIT: int = 120

    # Retrieval
    DEFAULT_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 1.1

    # Generation
    GENERATION_TEMPERATURE: float = 0.0
    GENERATION_MAX_TOKENS: int = 2048

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
