import os
import json
import logging
from pathlib import Path
from typing import Optional
from lxml import etree

from src.config import settings
from src.models import TrafficLawArticle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text_list: list[str]) -> str:
    if not text_list:
        return ""
    return " ".join(" ".join(text_list).split())


def parse_xml_file(filepath: Path) -> Optional[TrafficLawArticle]:
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()

        if root.findtext(".//META_ARTICLE/ETAT") != "VIGUEUR":
            return None

        article_id = root.findtext(".//META_COMMUN/ID")
        num = root.findtext(".//META_ARTICLE/NUM")

        content_text = ""
        bloc = root.find(".//BLOC_TEXTUEL")
        if bloc is not None:
            content_text = clean_text(list(bloc.itertext()))

        if not content_text:
            contenu = root.find(".//CONTENU")
            if contenu is not None:
                content_text = clean_text(list(contenu.itertext()))

        parents = []
        ctx = root.find(".//CONTEXTE")
        if ctx is not None:
            parents = [t.strip() for t in ctx.itertext() if t.strip()]

        context_str = " > ".join(parents) if parents else "Code de la Route"

        return TrafficLawArticle(
            id=article_id,
            article_number=num,
            content=content_text,
            context=context_str,
        )

    except etree.XMLSyntaxError:
        return None
    except ValueError:
        return None
    except Exception as e:
        logger.error(f"Error parsing {filepath.name}: {e}")
        return None


def process_directory(source_dir: Path) -> list[TrafficLawArticle]:
    articles = []
    if not source_dir.exists():
        return []

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".xml"):
                article = parse_xml_file(Path(root) / filename)
                if article:
                    articles.append(article)

    logger.info(f"Parsed {len(articles)} valid articles.")
    return articles


def main():
    articles = process_directory(settings.RAW_DATA_DIR)
    if not articles:
        return

    output = settings.PROCESSED_FILE
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in articles], f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(articles)} articles to {output}")


if __name__ == "__main__":
    main()
