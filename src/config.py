from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Provider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


PROVIDER_MODELS = {
    Provider.GEMINI: {
        "classifier": "models/gemini-3.1-flash-lite-preview",
        "generation": "models/gemini-3.1-flash-lite-preview",
        "embedding": "gemini-embedding-001",
    },
    Provider.OLLAMA: {
        "classifier": "qwen2.5:7b-instruct",
        "generation": "qwen2.5:7b-instruct",
        "embedding": "nomic-embed-text",
    },
}

# Native embedding dimensions per provider (must match the chosen embedding model).
PROVIDER_EMBEDDING_DIMENSIONS = {
    Provider.GEMINI: 3072,
    Provider.OLLAMA: 768,
}


class Settings(BaseSettings):

    GOOGLE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    LEGIFRANCE_CLIENT_ID: str = ""
    LEGIFRANCE_CLIENT_SECRET: str = ""
    PROVIDER: Provider = Provider.GEMINI

    # Ollama runtime
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_REQUEST_TIMEOUT: float = 120.0

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
    def CLASSIFIER_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["classifier"]

    @property
    def GENERATION_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["generation"]

    @property
    def EMBEDDING_MODEL(self) -> str:
        return PROVIDER_MODELS[self.PROVIDER]["embedding"]

    # Pinecone
    PINECONE_INDEX_NAME: str = "traffic-law-v1"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        return PROVIDER_EMBEDDING_DIMENSIONS[self.PROVIDER]

    # Indexing (batch)
    BATCH_SIZE: int = 5
    SLEEP_BETWEEN_BATCHES: int = 5
    MAX_RETRIES: int = 20
    RETRY_MIN_WAIT: int = 10
    RETRY_MAX_WAIT: int = 120

    # Query-time retry (interactive)
    QUERY_MAX_RETRIES: int = 3
    QUERY_RETRY_MIN_WAIT: float = 1.0
    QUERY_RETRY_MAX_WAIT: float = 8.0

    # Retrieval
    DEFAULT_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.5

    # Generation
    GENERATION_TEMPERATURE: float = 0.0
    GENERATION_MAX_TOKENS: int = 2048

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

HISTORY_WINDOW = 3
