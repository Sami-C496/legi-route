import json
import logging
import time
import chromadb
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception

from src.config import settings
from src.models import TrafficLawArticle
from src.providers import get_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

provider = get_provider()


def _is_retriable(exc):
    """Don't retry 400 errors (bad request), only rate limits and server errors."""
    exc_str = str(exc)
    if "400" in exc_str or "invalid_request" in exc_str:
        return False
    return True


@retry(
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=settings.RETRY_MIN_WAIT, max=settings.RETRY_MAX_WAIT),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def compute_embeddings(texts: list[str]) -> list[list[float]]:
    return provider.embed(texts, task_type="document")


def load_validated_data() -> list[TrafficLawArticle]:
    if not settings.PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {settings.PROCESSED_FILE}")

    with open(settings.PROCESSED_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [TrafficLawArticle(**item) for item in raw_data]


def main():
    logger.info("Starting indexing pipeline...")

    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
    collection = chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)

    articles = load_validated_data()
    logger.info(f"Loaded {len(articles)} articles.")

    total_new = 0

    for i in tqdm(range(0, len(articles), settings.BATCH_SIZE), desc="Indexing"):
        batch = articles[i : i + settings.BATCH_SIZE]

        existing = set(collection.get(ids=[a.id for a in batch], include=[])["ids"])
        new = [a for a in batch if a.id not in existing]

        if not new:
            continue

        embeddings = compute_embeddings([a.blob_for_embedding for a in new])

        collection.add(
            ids=[a.id for a in new],
            embeddings=embeddings,
            metadatas=[{
                "article_id": a.id,
                "num": a.article_number,
                "category": a.context,
                "content": a.content,
                "url": a.full_url,
            } for a in new],
            documents=[a.blob_for_embedding for a in new],
        )

        total_new += len(new)
        time.sleep(settings.SLEEP_BETWEEN_BATCHES)

    logger.info(f"Done. Indexed {total_new} new documents. Total: {collection.count()}")


if __name__ == "__main__":
    main()
