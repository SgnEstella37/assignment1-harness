"""Problem 12 (memory): persistent facts under memory_dir + optional recall prefetch."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

_FACTS_FILE = "facts.jsonl"

_DECIDE_SYSTEM = (
    "You decide whether an agent should load persistent personal memory "
    "before answering the user's message.\n"
    "Reply with ONLY one JSON object, no markdown:\n"
    '{"recall": true or false, "query": "<short search query if recall is true>"}\n'
    "Set recall=true when the user message depends on who they are, their team, "
    "preferences, cost center, nicknames, prior instructions, or other personal facts "
    "that may have been stored in an earlier session.\n"
    "Set recall=false for pure company-doc lookups with no personal context, or when "
    "the user is clearly asking you to store a new fact rather than use an old one.\n"
    "If recall=true, query should be a short keyword phrase for searching stored facts."
)


def facts_path(memory_dir: Path | str) -> Path:
    return Path(memory_dir) / _FACTS_FILE


def load_facts(memory_dir: Path | str | None) -> list[str]:
    if memory_dir is None:
        return []
    path = facts_path(memory_dir)
    if not path.exists():
        return []
    facts: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            fact = json.loads(ln).get("fact", "")
        except json.JSONDecodeError:
            fact = ln.strip()
        if fact:
            facts.append(str(fact))
    return facts


def _fact_key(fact: str) -> str | None:
    """Slot key for corrections, e.g. 'Employee number: 31908' -> 'employee number'."""
    text = (fact or "").strip()
    if ":" in text:
        return text.split(":", 1)[0].strip().lower() or None
    return None


def store_fact(memory_dir: Path | str, fact: str) -> str:
    path = Path(memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    store = facts_path(path)
    existing = load_facts(path)
    key = _fact_key(fact)
    if key:
        # Corrections replace earlier values in the same slot.
        existing = [f for f in existing if _fact_key(f) != key]
    existing.append(fact)
    with open(store, "w", encoding="utf-8") as f:
        for item in existing:
            f.write(json.dumps({"fact": item}, ensure_ascii=False) + "\n")
    return f"Remembered: {fact}"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def search_facts(facts: list[str], query: str) -> list[str]:
    if not facts:
        return []
    # Newest facts first so corrections win over earlier typos.
    facts_newest_first = list(reversed(facts))
    q = (query or "").strip().lower()
    if not q:
        return facts_newest_first
    # Substring hits first (still newest-first among ties).
    hits = [f for f in facts_newest_first if q in f.lower()]
    if hits:
        return hits
    # Token overlap ranking; break ties by recency (already newest-first).
    q_toks = _tokens(q)
    if not q_toks:
        return facts_newest_first
    scored: list[tuple[int, int, str]] = []
    for idx, f in enumerate(facts_newest_first):
        overlap = len(q_toks & _tokens(f))
        if overlap:
            scored.append((overlap, -idx, f))
    if scored:
        scored.sort(reverse=True)
        return [f for _, _, f in scored]
    # Nothing matched: return all so the agent can still use personal context.
    return facts_newest_first


def _dedupe_newest(facts: list[str]) -> list[str]:
    """Keep one fact per slot key. ``facts`` should be newest-first; first wins."""
    seen: set[str] = set()
    out: list[str] = []
    for fact in facts:
        key = _fact_key(fact)
        if key:
            if key in seen:
                continue
            seen.add(key)
        out.append(fact)
    return out


def format_recall(facts: list[str], *, empty_message: str) -> str:
    if not facts:
        return empty_message
    facts = _dedupe_newest(facts)
    return "Stored facts:\n- " + "\n- ".join(facts)


def recall(memory_dir: Path | str | None, query: str) -> str:
    if memory_dir is None:
        return "Memory is empty (no memory_dir)."
    facts = load_facts(memory_dir)
    if not facts:
        return "Memory is empty. No facts stored yet."
    hits = search_facts(facts, query)
    return format_recall(hits, empty_message="No matching facts found.")


def _parse_decision(text: str) -> tuple[bool, str]:
    raw = (text or "").strip()
    if not raw:
        return False, ""
    # Prefer a JSON object somewhere in the reply.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                do = bool(obj.get("recall", False))
                query = str(obj.get("query") or obj.get("q") or "").strip()
                return do, query
        except json.JSONDecodeError:
            pass
    lower = raw.lower()
    if re.search(r"\brecall\b\s*[:=]\s*true", lower) or re.search(
        r'"recall"\s*:\s*true', lower
    ):
        q_match = re.search(r'"query"\s*:\s*"([^"]*)"', raw)
        return True, (q_match.group(1).strip() if q_match else "")
    return False, ""


def decide_recall(
    aux_lm: Callable[..., Any],
    user_message: str,
    *,
    n_facts: int,
) -> tuple[bool, str]:
    """Ask aux_lm whether to prefetch recall for this session/message."""
    if n_facts <= 0:
        return False, ""
    reply = aux_lm(
        [
            {"role": "system", "content": _DECIDE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"There are {n_facts} stored personal fact(s) from earlier sessions.\n\n"
                    f"User message:\n{user_message}\n\n"
                    "Decide now. JSON only."
                ),
            },
        ]
    )
    text = reply if isinstance(reply, str) else str(reply)
    do, query = _parse_decision(text)
    if do and not query:
        # Fall back to a broad query so search_facts returns useful hits.
        query = user_message[:120]
    return do, query
