"""
Indexing Pipeline for Traffic Law RAG.

Handles the ingestion of processed JSON data into ChromaDB with:
- Idempotency checks (safe to restart without duplicates)
- Exponential backoff for API rate limits
- Batch processing with hard throttling for free tier
"""

import json
import logging
import time
import chromadb
from typing import List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from google import genai
from google.genai import types

from src.config import settings
from src.models import TrafficLawArticle

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- API Client ---
if not settings.GOOGLE_API_KEY:
    raise ValueError("Critical: GOOGLE_API_KEY is missing from environment variables.")

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


@retry(
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_exponential(
        multiplier=2,
        min=settings.RETRY_MIN_WAIT,
        max=settings.RETRY_MAX_WAIT
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def compute_embeddings_with_backoff(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings with robust retry logic.
    Task type RETRIEVAL_DOCUMENT optimizes vectors for storage/indexing.
    """
    try:
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        return [e.values for e in response.embeddings]

    except Exception as e:
        logger.error(f"Embedding API request failed: {e}")
        raise


def load_validated_data() -> List[TrafficLawArticle]:
    """Loads raw JSON data and validates it against the Pydantic schema."""
    input_file = settings.PROCESSED_FILE

    if not input_file.exists():
        raise FileNotFoundError(f"Source file not found at: {input_file}")

    logger.info(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [TrafficLawArticle(**item) for item in raw_data]


def main():
    """
    Main pipeline: Connect DB → Load Data → Batch (Check → Embed → Upsert).
    """
    logger.info("🚀 Starting Indexing Pipeline...")

    # 1. Initialize Vector Database
    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
    collection = chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)
    logger.info(f"Connected to ChromaDB collection: '{settings.COLLECTION_NAME}'")

    # 2. Load & Validate Data
    try:
        articles = load_validated_data()
        logger.info(f"Successfully loaded {len(articles)} articles.")
    except Exception as e:
        logger.critical(f"Data loading failed: {e}")
        return

    # 3. Batch Processing
    total_new_docs = 0

    for i in tqdm(range(0, len(articles), settings.BATCH_SIZE), desc="Processing Batches"):
        batch_articles = articles[i : i + settings.BATCH_SIZE]

        # Idempotency: check existing IDs before embedding
        batch_ids = [a.id for a in batch_articles]
        existing_records = collection.get(ids=batch_ids, include=[])
        existing_ids_set = set(existing_records['ids'])

        new_articles = [a for a in batch_articles if a.id not in existing_ids_set]

        if not new_articles:
            continue

        # Prepare for Chroma
        new_ids = [a.id for a in new_articles]
        new_docs = [a.blob_for_embedding for a in new_articles]
        new_metadatas = [{
            "article_id": a.id,
            "num": a.article_number,
            "category": a.context,
            "url": a.full_url
        } for a in new_articles]

        # Embed & Index
        embeddings = compute_embeddings_with_backoff(new_docs)

        collection.add(
            ids=new_ids,
            embeddings=embeddings,
            metadatas=new_metadatas,
            documents=new_docs
        )

        total_new_docs += len(new_articles)
        time.sleep(settings.SLEEP_BETWEEN_BATCHES)

    logger.info(f"Pipeline finished. Indexed {total_new_docs} new documents.")
    logger.info(f"Total collection size: {collection.count()} documents.")


if __name__ == "__main__":
    main()
