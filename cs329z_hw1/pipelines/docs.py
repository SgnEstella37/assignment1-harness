"""Problem 9 (search_docs): BM25 retrieval over company documents."""

from __future__ import annotations

from dataclasses import dataclass

from cs329z_hw1.pipelines.bm25 import BM25Index, build_bm25, search_bm25
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens

SNIPPET_MAX_TOKENS = 150
WINDOW_MAX_TOKENS = 1500


@dataclass
class DocIndex:
    docs: list[dict]  # {doc_id, title, text}
    by_id: dict[str, dict]
    bm25: BM25Index
    # For each doc, the raw text used for BM25 (same as doc["text"]).
    # Document ids in bm25 are positions into docs.


def build_doc_index(docs: list[dict]) -> DocIndex:
    docs = list(docs)
    by_id = {d["doc_id"]: d for d in docs}
    bm25 = build_bm25([d["text"] for d in docs])
    return DocIndex(docs=docs, by_id=by_id, bm25=bm25)


def _snippet_for(doc: dict, query: str) -> str:
    """Short excerpt that contains query terms when possible (<= 150 tokens)."""
    text = doc["text"]
    lines = text.splitlines() or [text]
    q_terms = set(tokenize(query))

    # Prefer a line that contains a query term.
    best_i = 0
    best_hits = -1
    for i, line in enumerate(lines):
        hits = len(q_terms & set(tokenize(line))) if q_terms else 0
        if hits > best_hits:
            best_hits = hits
            best_i = i

    # Grow a window of lines around best_i within the token budget.
    start = end = best_i
    snippet = lines[best_i]
    while count_tokens(snippet) > SNIPPET_MAX_TOKENS and len(snippet) > 20:
        # Trim from the ends of this single long line.
        cut = max(20, len(snippet) // 10)
        snippet = snippet[cut : len(snippet) - cut]
    if count_tokens(snippet) > SNIPPET_MAX_TOKENS:
        # Character fallback using ~4 chars/token from tokens.py convention.
        max_chars = SNIPPET_MAX_TOKENS * 4
        snippet = snippet[:max_chars]

    # Try to include neighboring lines while under budget.
    while True:
        expanded = False
        if start > 0:
            candidate = "\n".join(lines[start - 1 : end + 1])
            if count_tokens(candidate) <= SNIPPET_MAX_TOKENS:
                start -= 1
                snippet = candidate
                expanded = True
        if end + 1 < len(lines):
            candidate = "\n".join(lines[start : end + 2])
            if count_tokens(candidate) <= SNIPPET_MAX_TOKENS:
                end += 1
                snippet = candidate
                expanded = True
        if not expanded:
            break

    snippet = snippet.strip()
    if not snippet:
        snippet = (lines[0] if lines else text)[: SNIPPET_MAX_TOKENS * 4]
    return snippet


def search_docs(index: DocIndex, query: str, k: int = 5) -> list[dict]:
    if k <= 0:
        return []
    hits = search_bm25(index.bm25, query, k)
    results: list[dict] = []
    for doc_id, _score in hits:
        doc = index.docs[doc_id]
        results.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "snippet": _snippet_for(doc, query),
            }
        )
    return results


def read_doc(index: DocIndex, doc_id: str, start_line: int = 1) -> dict:
    doc = index.by_id.get(doc_id)
    if doc is None:
        raise LookupError(f"Unknown doc_id: {doc_id!r}")
    lines = doc["text"].splitlines()
    total = len(lines)
    if total == 0:
        raise ValueError(f"Document {doc_id!r} has no lines")
    if not isinstance(start_line, int) or start_line < 1 or start_line > total:
        raise ValueError(f"start_line {start_line} is outside 1..{total} for {doc_id}")

    end_line = start_line
    text = lines[start_line - 1]
    next_line = start_line + 1
    while next_line <= total:
        candidate = "\n".join(lines[start_line - 1 : next_line])
        if count_tokens(candidate) > WINDOW_MAX_TOKENS:
            break
        text = candidate
        end_line = next_line
        next_line += 1

    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total,
        "text": text,
    }
