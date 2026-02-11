"""
XML Parser for Légifrance LEGI Dataset.

Parses raw XML files from the DILA data dump, validates them against
the Pydantic schema, and outputs a clean JSON dataset.

Only articles with ETAT == "VIGUEUR" (currently enforced) are retained.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from lxml import etree

from src.models import TrafficLawArticle

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Paths ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "LEGI" / "TEXT"
    / "00" / "00" / "06" / "07" / "42"
    / "LEGITEXT000006074228" / "article"
) # This is the path to "code de la route" articles in the DILA dump.
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "code_route_articles.json"


def clean_text(text_list: List[str]) -> str:
    """
    Sanitizes and concatenates a list of text fragments.

    Args:
        text_list: A list of strings extracted from XML nodes.

    Returns:
        A single cleaned string with normalized whitespace.
    """
    if not text_list:
        return ""
    full_text = " ".join(text_list)
    return " ".join(full_text.split())


def parse_xml_file(filepath: Path) -> Optional[TrafficLawArticle]:
    """
    Parses a single Légifrance XML file into a validated TrafficLawArticle.

    Returns None if the article is not in VIGUEUR status, has empty content,
    or contains malformed XML.
    """
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()

        # 1. STATUS CHECK — Only currently enforced laws
        etat = root.findtext(".//META_ARTICLE/ETAT")
        if etat != "VIGUEUR":
            return None

        # 2. EXTRACT METADATA
        article_id = root.findtext(".//META_COMMUN/ID")
        num = root.findtext(".//META_ARTICLE/NUM")

        # 3. EXTRACT CONTENT
        # Strategy A: Standard structure (BLOC_TEXTUEL wrapper)
        content_text = ""
        bloc_textuel = root.find(".//BLOC_TEXTUEL")
        if bloc_textuel is not None:
            content_text = clean_text(bloc_textuel.itertext())

        # Strategy B: Fallback (direct CONTENU) if BLOC_TEXTUEL is missing
        if not content_text:
            logger.warning(f"No BLOC_TEXTUEL found in {filepath.name}. Attempting fallback extraction.")
            contenu_node = root.find(".//CONTENU")
            if contenu_node is not None:
                content_text = clean_text(contenu_node.itertext())

        # 4. RECONSTRUCT CONTEXT HIERARCHY
        parents = []
        contexte_node = root.find(".//CONTEXTE")
        if contexte_node is not None:
            for text_node in contexte_node.itertext():
                clean_t = text_node.strip()
                if clean_t:
                    parents.append(clean_t)

        full_path_list = [p.strip() for p in parents]
        context_str = " > ".join(full_path_list) if full_path_list else "Code de la Route"

        # 5. VALIDATION via Pydantic (rejects empty/short content)
        article = TrafficLawArticle(
            id=article_id,
            article_number=num,
            content=content_text,
            context=context_str
        )
        return article

    except etree.XMLSyntaxError:
        logger.error(f"XML Syntax Error in file: {filepath}")
        return None
    except ValueError:
        # Pydantic validation failure (e.g., content too short)
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {filepath.name}: {e}", exc_info=True)
        return None


def process_directory(source_dir: Path) -> List[TrafficLawArticle]:
    """Recursively processes a directory of XML files."""
    articles = []
    file_count = 0

    if not source_dir.exists():
        logger.critical(f"Source directory does not exist: {source_dir}")
        return []

    logger.info(f"Starting ingestion from: {source_dir}")

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".xml"):
                file_count += 1
                filepath = Path(root) / filename

                article = parse_xml_file(filepath)
                if article:
                    articles.append(article)

                if file_count % 1000 == 0:
                    logger.info(f"Processed {file_count} files... Collected {len(articles)} articles.")

    logger.info(f"Ingestion complete. Processed {file_count} files.")
    logger.info(f"Total valid articles retained: {len(articles)}")
    return articles


def main():
    """Entry point for the parsing script."""
    articles = process_directory(RAW_DATA_DIR)

    if articles:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                data_to_save = [article.model_dump() for article in articles]
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved {len(articles)} articles to {OUTPUT_FILE}")

        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
    else:
        logger.warning("No articles were found or processed. JSON file was not created.")


if __name__ == "__main__":
    main()
