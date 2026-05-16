import re
import unicodedata
from typing import Optional

from bs4 import BeautifulSoup

from app.core.logging import get_logger

logger = get_logger(__name__)

# Tags whose inner text we promote to a labelled line so the chunker
# can treat them as natural split boundaries.
_HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_EMPHASIS_TAGS = {"strong", "b"}


def clean_html(raw_html: str) -> str:

    if not raw_html or not raw_html.strip():
        return ""

    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception as exc:
        logger.warning("html_parse_failed", error=str(exc))
        # Fallback: strip all tags with a simple regex
        return _strip_tags_regex(raw_html)

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "head", "nav", "footer"]):
        tag.decompose()

    # Replace list items with bullet prefix
    for li in soup.find_all("li"):
        li.insert_before("• ")
        li.insert_after("\n")

    # Promote headers to labelled lines
    for tag in soup.find_all(_HEADER_TAGS):
        tag.string = f"\n{tag.get_text(strip=True).upper()}:\n"

    # Promote emphasis/bold to inline uppercase cues
    for tag in soup.find_all(_EMPHASIS_TAGS):
        text = tag.get_text(strip=True)
        if text:
            tag.replace_with(f"{text}: ")

    # Extract text — use newline as block separator
    text = soup.get_text(separator="\n")

    # Normalise whitespace
    text = _normalise_whitespace(text)

    # Unicode NFC normalisation (handles decomposed diacritics, etc.)
    text = unicodedata.normalize("NFC", text)

    return text.strip()


def build_chunk_document(row: dict) -> str:

    tags = row.get("Tags", "") or ""
    tags_text = f"Tags: {tags}\n" if tags.strip() else ""

    header = (
        f"Job Title: {row.get('Job Title', '')}\n"
        f"Company: {row.get('Company Name', '')}\n"
        f"Category: {row.get('Job Category', '')}\n"
        f"Level: {row.get('Job Level', '')}\n"
        f"Location: {row.get('Job Location', '')}\n"
        f"Published: {row.get('Publication Date', '')}\n"
        f"{tags_text}"
        f"\nDescription:\n"
    )

    description = clean_html(row.get("Job Description", "") or "")
    return header + description


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and limit consecutive newlines to two."""
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse vertical whitespace (>2 newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text


def _strip_tags_regex(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return _normalise_whitespace(text)


def truncate_text(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    # Find last space before the limit
    cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut] + "…"
