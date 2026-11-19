"""Problem 1 (priority): classify one email with a single LM call."""

from __future__ import annotations

import re

from cs329z_hw1.types import Email, LMCallable

CATEGORIES = ("urgent", "normal", "ignore")

SYSTEM_PROMPT = """\
You classify one email for its recipient (the people listed in To).
Judge from that person's point of view at the time the email was sent.
The question is what they should do with the message, not how important
the topic is to the company.

Pick exactly one label: urgent, normal, or ignore.

Definitions:
- urgent: the recipient must act or decide soon. BOTH must be true:
  (1) the email asks them to do something, or reports a live problem they
      are responsible for (outage, failed payment, escalation);
  (2) there is time pressure: same day, next working day, or a stated
      deadline at most two working days away (Mon-Fri from the email date).
- normal: legitimate work or personal mail for this recipient or a small
  group, with no deadline in the next two working days. Includes status
  updates, FYIs, ordinary questions, documents for review, later scheduling,
  ongoing discussion replies, and personal notes from people they know.
- ignore: mail they can leave unread at no cost — advertising/spam,
  newsletters, announcements to large employee lists, and automated
  notifications that ask nothing of this recipient.

Decision rules:
- Wording alone does not decide the label. "URGENT" in the subject is not
  enough if nothing is asked of the recipient. A calm "please sign by 3pm
  today" is urgent.
- A request with a deadline more than two working days away, or with no
  deadline, is normal.
- Mass mailings and ads are ignore even if they say "act now" or "important".
- An automated notice that asks this specific recipient to act is normal,
  or urgent if the deadline is within two working days.
- A short personal/social message from someone they know is normal, not ignore.

Reply in exactly this format and nothing else:
category: <urgent|normal|ignore>
reason: <one short sentence>
"""


def _format_email(email: Email) -> str:
    to_list = email.get("to") or []
    to_text = ", ".join(to_list) if isinstance(to_list, list) else str(to_list)
    return (
        f"From: {email.get('from', '')}\n"
        f"To: {to_text}\n"
        f"Date: {email.get('date', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"\n"
        f"{email.get('body', '')}"
    )


def _find_labels(text: str) -> list[str]:
    """Return label words found in text, in order of first appearance."""
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    for label in CATEGORIES:
        for match in re.finditer(rf"\b{label}\b", lowered):
            hits.append((match.start(), label))
    hits.sort(key=lambda item: item[0])
    # unique, preserving first-seen order
    seen: list[str] = []
    for _, label in hits:
        if label not in seen:
            seen.append(label)
    return seen


def _parse_reason(reply: str, category: str) -> str:
    """Pull a non-empty reason out of the model reply."""
    # Prefer an explicit "reason:" line.
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("reason:"):
            reason = stripped.split(":", 1)[1].strip()
            if reason:
                return reason

    # Otherwise use the reply with label tokens removed, cleaned up.
    cleaned = reply.strip()
    cleaned = re.sub(r"(?im)^\s*category\s*:\s*\w+\s*$", "", cleaned)
    cleaned = re.sub(rf"(?i)\b{category}\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-\n\t")
    if cleaned:
        return cleaned
    return f"Model labeled the email as {category}."


def parse_priority_reply(reply: str) -> dict:
    """Turn a free-form model string into the required result dict."""
    if not isinstance(reply, str):
        reply = str(reply) if reply is not None else ""

    labels = _find_labels(reply)
    if len(labels) == 1:
        category = labels[0]
        reason = _parse_reason(reply, category)
        return {"category": category, "reason": reason}

    # No label, or more than one distinct label with no clear single choice.
    # Prefer an explicit "category:" line when present.
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("category:"):
            value = stripped.split(":", 1)[1].strip().lower()
            for label in CATEGORIES:
                if re.search(rf"\b{label}\b", value):
                    return {"category": label, "reason": _parse_reason(reply, label)}

    if len(labels) == 0:
        reason = reply.strip() or "Model reply stated no priority label."
        return {"category": "unknown", "reason": reason}

    # Multiple labels mentioned; do not guess.
    reason = reply.strip() or "Model reply mentioned multiple labels."
    return {"category": "unknown", "reason": reason}


def classify_priority(email: Email, lm: LMCallable) -> dict:
    """Classify one email; never raises; never modifies ``email``."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Classify this email.\n\n" + _format_email(email),
        },
    ]
    try:
        reply = lm(messages)
    except Exception as exc:  # pragma: no cover - contract: still return a dict
        return {
            "category": "unknown",
            "reason": f"Model call failed: {exc}",
        }
    return parse_priority_reply(reply if isinstance(reply, str) else str(reply))
