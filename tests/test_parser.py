"""
Tests for the XML parser (parser.py).

Strategy: We create minimal synthetic XML strings that mimic the Légifrance
LEGI format. This makes the tests self-contained — no dependency on the
real 2GB XML dump.
"""

import pytest
from pathlib import Path
from lxml import etree

from src.ingestion.parser import parse_xml_file, clean_text, process_directory


# --- Helpers ---

def write_xml(tmp_path: Path, filename: str, xml_content: str) -> Path:
    """Write an XML string to a temporary file and return its path."""
    filepath = tmp_path / filename
    filepath.write_text(xml_content, encoding="utf-8")
    return filepath


# --- XML Fixtures ---

VALID_ARTICLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ARTICLE>
    <META>
        <META_COMMUN>
            <ID>LEGIARTI000006841575</ID>
        </META_COMMUN>
        <META_ARTICLE>
            <ETAT>VIGUEUR</ETAT>
            <NUM>R413-17</NUM>
        </META_ARTICLE>
    </META>
    <BLOC_TEXTUEL>
        <CONTENU>
            <p>Sur les autoroutes, la vitesse est limitée à 130 km/h.</p>
        </CONTENU>
    </BLOC_TEXTUEL>
    <CONTEXTE>
        <TEXTE>Code de la route</TEXTE>
        <TEXTE>Partie réglementaire</TEXTE>
        <TEXTE>Livre IV</TEXTE>
    </CONTEXTE>
</ARTICLE>
"""

ABROGATED_ARTICLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ARTICLE>
    <META>
        <META_COMMUN><ID>LEGIARTI000099999999</ID></META_COMMUN>
        <META_ARTICLE>
            <ETAT>ABROGE</ETAT>
            <NUM>R999-1</NUM>
        </META_ARTICLE>
    </META>
    <BLOC_TEXTUEL>
        <CONTENU><p>Cet article est abrogé.</p></CONTENU>
    </BLOC_TEXTUEL>
</ARTICLE>
"""

EMPTY_CONTENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ARTICLE>
    <META>
        <META_COMMUN><ID>LEGIARTI000000000001</ID></META_COMMUN>
        <META_ARTICLE>
            <ETAT>VIGUEUR</ETAT>
            <NUM>R000-1</NUM>
        </META_ARTICLE>
    </META>
    <BLOC_TEXTUEL>
        <CONTENU></CONTENU>
    </BLOC_TEXTUEL>
</ARTICLE>
"""

FALLBACK_CONTENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ARTICLE>
    <META>
        <META_COMMUN><ID>LEGIARTI000000000002</ID></META_COMMUN>
        <META_ARTICLE>
            <ETAT>VIGUEUR</ETAT>
            <NUM>L123-4</NUM>
        </META_ARTICLE>
    </META>
    <CONTENU>
        <p>Contenu direct sans BLOC_TEXTUEL.</p>
    </CONTENU>
    <CONTEXTE>
        <TEXTE>Code de la route</TEXTE>
    </CONTEXTE>
</ARTICLE>
"""


# ============================================================
# clean_text
# ============================================================

class TestCleanText:
    """Unit tests for the text sanitization function."""

    def test_joins_fragments(self):
        assert clean_text(["Hello", "World"]) == "Hello World"

    def test_normalizes_whitespace(self):
        assert clean_text(["  too   many\t\tspaces\n\n"]) == "too many spaces"

    def test_empty_list_returns_empty_string(self):
        assert clean_text([]) == ""

    def test_strips_leading_trailing(self):
        result = clean_text(["  padded  "])
        assert not result.startswith(" ")
        assert not result.endswith(" ")


# ============================================================
# parse_xml_file — Valid Articles
# ============================================================

class TestParseValidArticle:
    """Tests for successful parsing of well-formed XML."""

    def test_parses_vigueur_article(self, tmp_path):
        filepath = write_xml(tmp_path, "valid.xml", VALID_ARTICLE_XML)
        article = parse_xml_file(filepath)

        assert article is not None
        assert article.id == "LEGIARTI000006841575"
        assert article.article_number == "R413-17"
        assert "130 km/h" in article.content

    def test_extracts_context_hierarchy(self, tmp_path):
        filepath = write_xml(tmp_path, "valid.xml", VALID_ARTICLE_XML)
        article = parse_xml_file(filepath)

        assert "Code de la route" in article.context
        assert "Partie réglementaire" in article.context
        assert "Livre IV" in article.context
        assert " > " in article.context

    def test_fallback_to_contenu_when_no_bloc_textuel(self, tmp_path):
        """Parser should try <CONTENU> if <BLOC_TEXTUEL> is missing."""
        filepath = write_xml(tmp_path, "fallback.xml", FALLBACK_CONTENT_XML)
        article = parse_xml_file(filepath)

        assert article is not None
        assert "Contenu direct" in article.content


# ============================================================
# parse_xml_file — Rejection Cases
# ============================================================

class TestParseRejection:
    """
    Tests for articles that should be filtered out.
    Critical: a legal RAG must NEVER index repealed laws.
    """

    def test_rejects_abrogated_article(self, tmp_path):
        """ETAT != VIGUEUR → must return None (not indexed)."""
        filepath = write_xml(tmp_path, "abrogated.xml", ABROGATED_ARTICLE_XML)
        result = parse_xml_file(filepath)
        assert result is None

    def test_rejects_empty_content(self, tmp_path):
        """Empty content fails Pydantic validation → returns None."""
        filepath = write_xml(tmp_path, "empty.xml", EMPTY_CONTENT_XML)
        result = parse_xml_file(filepath)
        assert result is None

    def test_handles_malformed_xml_gracefully(self, tmp_path):
        """Corrupted XML should not crash the pipeline."""
        filepath = write_xml(tmp_path, "broken.xml", "<NOT VALID XML<<<>")
        result = parse_xml_file(filepath)
        assert result is None

    def test_handles_missing_file_gracefully(self):
        result = parse_xml_file(Path("/nonexistent/file.xml"))
        assert result is None


# ============================================================
# process_directory
# ============================================================

class TestProcessDirectory:
    """Integration test: end-to-end directory processing."""

    def test_processes_mixed_directory(self, tmp_path):
        """A directory with 1 valid + 1 abrogated should yield exactly 1 article."""
        write_xml(tmp_path, "valid.xml", VALID_ARTICLE_XML)
        write_xml(tmp_path, "old.xml", ABROGATED_ARTICLE_XML)
        write_xml(tmp_path, "readme.txt", "not an xml file")

        articles = process_directory(tmp_path)
        assert len(articles) == 1
        assert articles[0].article_number == "R413-17"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        articles = process_directory(tmp_path)
        assert articles == []

    def test_nonexistent_directory_returns_empty_list(self):
        articles = process_directory(Path("/does/not/exist"))
        assert articles == []
