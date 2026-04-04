import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from src.config import settings
from src.models import TrafficLawArticle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"
API_BASE = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"
CODE_DE_LA_ROUTE_ID = "LEGITEXT000006074228"


def get_token() -> str:
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": settings.LEGIFRANCE_CLIENT_ID,
        "client_secret": settings.LEGIFRANCE_CLIENT_SECRET,
        "scope": "openid",
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["access_token"]


def _post(path: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _collect_ids(node: dict, ids: list) -> None:
    for art in node.get("articles", []):
        if art.get("etat") == "VIGUEUR":
            ids.append(art["id"])
    for section in node.get("sections", []):
        _collect_ids(section, ids)


def get_vigueur_ids(token: str) -> list[str]:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    toc = _post("/consult/code/tableMatieres", {"textId": CODE_DE_LA_ROUTE_ID, "date": date}, token)
    ids: list[str] = []
    _collect_ids(toc, ids)
    return ids


def _build_context(article: dict) -> str:
    ctx = article.get("context") or {}
    parts = []
    for entry in (ctx.get("titreTxt") or [])[:1]:
        parts.append(entry.get("titre", ""))
    for entry in ctx.get("titresTM") or []:
        titre = entry.get("titre", "")
        if titre:
            parts.append(titre)
    return " > ".join(p for p in parts if p) or "Code de la Route"


def fetch_article(article_id: str, token: str) -> Optional[TrafficLawArticle]:
    try:
        res = _post("/consult/getArticle", {"id": article_id}, token)
        a = res.get("article", {})
        content = " ".join((a.get("texte") or "").split())
        if len(content) < 5:
            return None
        return TrafficLawArticle(
            id=a["id"],
            article_number=a.get("num") or "",
            content=content,
            context=_build_context(a),
        )
    except Exception as e:
        logger.warning(f"Failed to fetch {article_id}: {e}")
        return None


def main():
    token = get_token()

    vigueur_ids = get_vigueur_ids(token)
    logger.info(f"{len(vigueur_ids)} VIGUEUR articles.")

    existing: dict[str, dict] = {}
    if settings.PROCESSED_FILE.exists():
        with open(settings.PROCESSED_FILE, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["id"]] = item

    vigueur_set = set(vigueur_ids)
    new_ids = [aid for aid in vigueur_ids if aid not in existing]
    logger.info(f"{len(existing)} existing, {len(new_ids)} to fetch.")

    articles = {aid: data for aid, data in existing.items() if aid in vigueur_set}

    for i, article_id in enumerate(new_ids):
        article = fetch_article(article_id, token)
        if article:
            articles[article_id] = article.model_dump()
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(new_ids)} fetched")
        time.sleep(0.2)

    output = list(articles.values())
    settings.PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(output)} articles written to {settings.PROCESSED_FILE}")


if __name__ == "__main__":
    main()
