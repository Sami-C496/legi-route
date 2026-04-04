import json
import logging
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
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


def get_or_create_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating index '{settings.PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )
        # Wait for index to be ready
        while not pc.describe_index(settings.PINECONE_INDEX_NAME).status.get("ready"):
            time.sleep(2)

    return pc.Index(settings.PINECONE_INDEX_NAME)


def main():
    logger.info("Starting indexing pipeline...")

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = get_or_create_index(pc)

    articles = load_validated_data()
    logger.info(f"Loaded {len(articles)} articles.")

    current_ids = {a.id for a in articles}

    logger.info("Fetching existing IDs from Pinecone...")
    pinecone_ids = set()
    for ids_batch in index.list():
        pinecone_ids.update(ids_batch)

    new_articles = [a for a in articles if a.id not in pinecone_ids]
    deleted_ids = list(pinecone_ids - current_ids)

    logger.info(f"{len(new_articles)} new, {len(deleted_ids)} to delete.")

    if deleted_ids:
        for i in range(0, len(deleted_ids), 1000):
            index.delete(ids=deleted_ids[i : i + 1000])
        logger.info(f"Deleted {len(deleted_ids)} articles.")

    if not new_articles:
        stats = index.describe_index_stats()
        logger.info(f"Nothing to index. Total vectors: {stats.total_vector_count}")
        return

    total_new = 0

    for i in tqdm(range(0, len(new_articles), settings.BATCH_SIZE), desc="Indexing"):
        batch = new_articles[i : i + settings.BATCH_SIZE]

        embeddings = compute_embeddings([a.blob_for_embedding for a in batch])

        vectors = [
            (
                a.id,
                emb,
                {
                    "article_id": a.id,
                    "num": a.article_number,
                    "category": a.context,
                    "content": a.content,
                    "url": a.full_url,
                },
            )
            for a, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        total_new += len(batch)
        time.sleep(settings.SLEEP_BETWEEN_BATCHES)

    stats = index.describe_index_stats()
    logger.info(f"Done. Indexed {total_new} new documents. Total: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
