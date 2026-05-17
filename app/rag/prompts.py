from __future__ import annotations

from typing import Any, Dict, List

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise job search assistant. Your role is to help users
find relevant job listings based on their query.

Rules:
1. Only use information from the provided job listings context.
2. Do not invent job details, companies, or requirements not present in the context.
3. If the context does not contain relevant information for the query, explicitly say so.
4. Be concise and structured. Avoid filler phrases.
5. When mentioning jobs, always include: job title, company name, location, and experience level.
6. Cite source numbers (e.g., [1], [3]) when referencing specific jobs."""


# ── Context formatting ─────────────────────────────────────────────────────────

def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a numbered context string.

    Each chunk is rendered with its most decision-relevant metadata fields
    at the top (title, company, location, level) followed by the matched
    text. The ``---`` separator gives the LLM a clean chunk boundary.

    Args:
        retrieved_chunks: List of result dicts from the retriever.
            Each must have ``metadata`` and ``text`` keys.

    Returns:
        A formatted string ready to embed in the answer synthesis prompt.
    """
    if not retrieved_chunks:
        return "No relevant job listings found."

    lines: List[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        lines.append(f"[{idx}] {meta.get('job_title', 'Unknown Title')}")
        lines.append(f"    Company: {meta.get('company_name', 'Unknown')}")
        lines.append(f"    Location: {meta.get('job_location', 'Unknown')}")
        lines.append(f"    Level: {meta.get('job_level', 'Unknown')}")
        lines.append(f"    Category: {meta.get('job_category', 'Unknown')}")
        if meta.get("tags"):
            lines.append(f"    Tags: {meta['tags']}")
        lines.append("")
        lines.append(f"    {chunk.get('text', '')[:600]}")
        lines.append("---")

    return "\n".join(lines)


def build_answer_prompt(query: str, context: str) -> str:
    """
    Construct the user-turn prompt for answer synthesis.

    Combines the user query with the formatted retrieval context into
    a single prompt. The explicit format request at the end keeps the
    model's output predictable and parseable.

    Args:
        query: The original user search query.
        context: Formatted context string from ``format_context``.

    Returns:
        A complete user-turn prompt string.
    """
    return f"""User Query: {query}

Retrieved Job Listings:
{context}

Based on the job listings above, provide a helpful answer to the user's query.
- Summarise the most relevant findings.
- Mention specific companies, job titles, locations, and experience levels.
- Cite listings by their number (e.g., [1], [2]).
- If no listings are relevant, say so clearly.
- Be concise (3–5 sentences or a short structured list).
- Do not make up any information not present in the listings above."""