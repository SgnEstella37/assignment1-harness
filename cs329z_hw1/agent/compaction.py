"""Problem 11 (compaction): keep every LM call within context_budget via aux_lm summaries."""

from __future__ import annotations

from typing import Any, Callable

from cs329z_hw1.tokens import count_message_tokens, count_tokens

SUMMARY_SYSTEM = (
    "You compress conversation history for an AI agent that will continue the same task.\n"
    "Preserve ALL of the following when present:\n"
    "- User goals, constraints, preferences, and requirements (budgets, locations, deadlines, names)\n"
    "- Important numbers, dates, IDs, and decisions\n"
    "- Outcomes of tools that still matter for later turns\n"
    "Omit raw tool-call markup, long repeated filler, and bulky dumps once their outcome is noted.\n"
    "Write a dense plain-text summary. Do not use tool-call syntax."
)

_HARD_NOTE = (
    "\n...\n[truncated to fit the context budget; head and tail kept]\n...\n"
)


def hard_truncate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    note = _HARD_NOTE
    avail = max_chars - len(note)
    if avail < 32:
        return text[:max_chars]
    head = (avail * 2) // 3
    tail = avail - head
    out = text[:head] + note + text[-tail:]
    while count_tokens(out) > max_tokens and head > 16:
        head = max(16, head - 32)
        tail = max(0, avail - head)
        out = text[:head] + note + text[-tail:]
    return out


def _format_for_summary(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        # Cap each piece so the summarizer input itself stays manageable.
        if count_tokens(content) > 4000:
            content = hard_truncate(content, 4000)
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def summarize_with_lm(aux_lm: Callable[..., Any], messages: list[dict]) -> str:
    """Ask aux_lm to summarize ``messages``. Never call the agent lm here."""
    if not messages:
        return "(empty)"
    blob = _format_for_summary(messages)
    reply = aux_lm(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Summarize this earlier conversation for the agent.\n\n" + blob
                ),
            },
        ]
    )
    text = reply if isinstance(reply, str) else str(reply)
    text = text.strip() or "(summary empty)"
    return text


def _aux_from_config(config, lm) -> Callable[..., Any]:
    if config.aux_lm is not None:
        return config.aux_lm
    return lm


def _message_budget(budget: int, reserved: int) -> int:
    return max(64, budget - reserved)


def fit_messages(
    messages: list[dict],
    *,
    budget: int,
    turn_start: int,
    aux_lm: Callable[..., Any],
) -> list[dict]:
    """Return a message list that fits ``budget``, preserving system + current turn.

    ``turn_start`` is the index of the user message that began the active send.
    Older turns may be replaced by an aux_lm summary. Oversized single messages
    are summarized (then hard-truncated if still too long).
    """
    if not messages:
        return messages
    if count_message_tokens(messages) <= budget:
        return list(messages)

    system = dict(messages[0])
    turn_start = max(1, min(turn_start, len(messages)))
    prefix = [dict(m) for m in messages[1:turn_start]]
    suffix = [dict(m) for m in messages[turn_start:]]

    system_tokens = count_message_tokens([system])
    # Reserve room for system + the current user message (first of suffix).
    current = suffix[0] if suffix else {"role": "user", "content": ""}
    current_tokens = count_message_tokens([current])
    reserved = system_tokens + current_tokens + 32

    def shrink_content(content: str, cap: int) -> str:
        if count_tokens(content) <= cap:
            return content
        # LLM-assisted compression first.
        summary = summarize_with_lm(
            aux_lm, [{"role": "user", "content": content}]
        )
        summary = f"[Compressed tool/context output]\n{summary}"
        if count_tokens(summary) <= cap:
            return summary
        return hard_truncate(summary, cap)

    # Shrink each suffix message so no single blob can blow the budget.
    per_msg_cap = max(256, budget // 2)
    for i, m in enumerate(suffix):
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
            m["content"] = content
        # Keep the current user message verbatim whenever possible.
        if i == 0:
            if count_tokens(content) > max(256, budget - system_tokens - 64):
                m["content"] = hard_truncate(
                    content, max(256, budget - system_tokens - 64)
                )
            continue
        if count_tokens(content) > per_msg_cap:
            m["content"] = shrink_content(content, per_msg_cap)

    out: list[dict]
    if prefix:
        summary = summarize_with_lm(aux_lm, prefix)
        # Keep the summary itself within a fraction of the budget.
        summary_cap = max(256, budget // 3)
        if count_tokens(summary) > summary_cap:
            summary = hard_truncate(summary, summary_cap)
        out = [
            system,
            {
                "role": "user",
                "content": (
                    "[Conversation summary so far — retain these facts]\n" + summary
                ),
            },
            {
                "role": "assistant",
                "content": "Understood. I will keep those points in mind.",
            },
        ] + suffix
    else:
        out = [system] + suffix

    if count_message_tokens(out) <= budget:
        return out

    # Still over: drop/compress older in-turn tool traffic, keep current user.
    # Walk from the oldest after current user, compressing large contents.
    guard = 0
    while count_message_tokens(out) > budget and guard < 32:
        guard += 1
        # Find the largest message that is not system and not the current user.
        current_idx = 1
        for i, m in enumerate(out):
            if i == 0:
                continue
            if m.get("content") == current.get("content") and m.get("role") == "user":
                current_idx = i
                break
        candidates = [
            i
            for i in range(len(out))
            if i not in (0, current_idx)
        ]
        if not candidates:
            # Last resort: hard-truncate current user (should be rare).
            out[current_idx]["content"] = hard_truncate(
                str(out[current_idx].get("content", "")),
                max(64, budget - system_tokens - 16),
            )
            break
        largest = max(candidates, key=lambda i: count_tokens(str(out[i].get("content", ""))))
        content = str(out[largest].get("content", ""))
        need = count_message_tokens(out) - budget
        new_cap = max(64, count_tokens(content) - need - 8)
        if new_cap >= count_tokens(content):
            new_cap = max(64, count_tokens(content) // 2)
        out[largest]["content"] = shrink_content(content, new_cap)

    # Absolute last resort: hard truncate everything except system + current user.
    if count_message_tokens(out) > budget:
        kept = [out[0]]
        # Find current user again
        cur = None
        for m in out[1:]:
            if m.get("role") == "user" and m.get("content") == current.get("content"):
                cur = m
                break
        if cur is None and len(out) > 1:
            cur = out[1]
        # Keep a short summary of the rest if any.
        others = [m for m in out[1:] if m is not cur]
        if others:
            summary = summarize_with_lm(aux_lm, others)
            summary = hard_truncate(summary, max(64, budget // 4))
            kept.append(
                {
                    "role": "user",
                    "content": "[Conversation summary so far]\n" + summary,
                }
            )
            kept.append(
                {
                    "role": "assistant",
                    "content": "Understood.",
                }
            )
        if cur is not None:
            # Ensure current user fits in remaining space.
            rem = budget - count_message_tokens(kept) - 8
            c = dict(cur)
            c["content"] = hard_truncate(str(c.get("content", "")), max(64, rem))
            kept.append(c)
        # Append a few most recent non-user-current messages if space remains.
        for m in reversed(out):
            if m is cur or m is out[0]:
                continue
            if m in kept:
                continue
            candidate = kept + [m]
            if count_message_tokens(candidate) <= budget:
                kept.append(m)
            if count_message_tokens(kept) > budget * 0.95:
                break
        out = kept

    # Final hard clamp if still slightly over.
    while count_message_tokens(out) > budget and len(out) > 2:
        # Remove the message before the last (keep system and last).
        if len(out) <= 3:
            out[-1]["content"] = hard_truncate(
                str(out[-1].get("content", "")),
                max(32, budget - count_message_tokens(out[:-1]) - 8),
            )
            break
        # Prefer removing middle tool results
        out.pop(2)

    if count_message_tokens(out) > budget:
        # Extreme clamp: system + truncated current user only.
        rem = max(32, budget - system_tokens - 8)
        out = [
            system,
            {
                "role": "user",
                "content": hard_truncate(str(current.get("content", "")), rem),
            },
        ]

    return out
