"""Problem 2 (daily_digest): morning summary over one day's emails."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

RANK = {"urgent": 0, "normal": 1, "ignore": 2, "unknown": 1}

DIGEST_SYSTEM = """\
You write a short morning email digest for a busy employee.
You will receive a filtered list of emails already classified as urgent or normal.
Ignore-pile mail has already been removed; do not invent or mention spam,
ads, newsletters, or unrelated vendor promotions.

Requirements:
- Write a concise morning summary in plain prose (short bullets are fine).
- At most 250 words. Stay well under 300 words.
- Cover every urgent item. For each urgent email, name the company, project,
  site, or other proper noun it is about so a reader can recognize it.
- Mention normal items briefly if space allows; urgent items come first.
- Do not invent emails that are not in the list.
- Return only the digest text, with no preamble or closing chatty lines.
"""


def _effective_category(category: str) -> str:
    if category == "unknown":
        return "normal"
    if category in RANK:
        return category
    return "normal"


def _email_blurb(email: Email, category: str) -> str:
    to_list = email.get("to") or []
    to_text = ", ".join(to_list) if isinstance(to_list, list) else str(to_list)
    body = (email.get("body") or "").strip()
    if len(body) > 800:
        body = body[:800] + "…"
    return (
        f"[{category.upper()}] id={email.get('id', '')} thread={email.get('thread_id', '')}\n"
        f"From: {email.get('from', '')}\n"
        f"To: {to_text}\n"
        f"Date: {email.get('date', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"{body}"
    )


def _classify_one(email: Email, lm: LMCallable) -> tuple[Email, str]:
    result = classify_priority(email, lm)
    category = _effective_category(result.get("category", "unknown"))
    return email, category


def _pick_per_thread(labeled: list[tuple[Email, str]]) -> list[tuple[Email, str]]:
    """Keep one email per thread: prefer urgent, then later messages."""
    best: dict[str, tuple[Email, str]] = {}
    order: list[str] = []
    for email, category in labeled:
        thread = email.get("thread_id") or email.get("id") or str(id(email))
        if thread not in best:
            order.append(thread)
            best[thread] = (email, category)
            continue
        prev_email, prev_cat = best[thread]
        if RANK[category] < RANK[prev_cat]:
            best[thread] = (email, category)
        elif RANK[category] == RANK[prev_cat]:
            # Same urgency: keep the later one by date, else the newer in input order.
            if (email.get("date") or "") >= (prev_email.get("date") or ""):
                best[thread] = (email, category)
    kept = [best[t] for t in order]
    kept.sort(key=lambda item: (RANK[item[1]], item[0].get("date") or ""))
    return kept


def _write_digest(kept: list[tuple[Email, str]], lm: LMCallable) -> str:
    if not kept:
        return "No action items for this morning."

    blocks = [_email_blurb(email, category) for email, category in kept]
    user = (
        "Write this morning's digest from the emails below.\n"
        "Remember: at most 250 words; name the key proper nouns in every urgent item.\n\n"
        + "\n\n-----\n\n".join(blocks)
    )
    reply = lm(
        [
            {"role": "system", "content": DIGEST_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    if not isinstance(reply, str):
        reply = str(reply) if reply is not None else ""
    text = reply.strip()
    return text if text else "No action items for this morning."


def make_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    """Build a morning digest string; never modifies ``emails``."""
    if not emails:
        return "No emails to summarize for this day."

    labeled: list[tuple[Email, str]] = []
    # Classify concurrently; calls are independent.
    max_workers = min(8, max(1, len(emails)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_classify_one, email, lm) for email in emails]
        for fut in as_completed(futures):
            labeled.append(fut.result())

    # Restore input order before thread collapse so behavior is stable.
    index = {id(email): i for i, email in enumerate(emails)}
    labeled.sort(key=lambda item: index.get(id(item[0]), 0))

    kept = [
        (email, category)
        for email, category in _pick_per_thread(labeled)
        if category != "ignore"
    ]
    return _write_digest(kept, lm)
