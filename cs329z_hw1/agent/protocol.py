"""Problem 6 (tool_protocol): text format for tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

from cs329z_hw1.types import ParsedResponse, ToolCall

BEGIN = "@@@TOOL_CALL_BEGIN@@@"
END = "@@@TOOL_CALL_END@@@"

# DeepSeek (and similar) sometimes emit alternate markup instead of @@@.
_TOOL_CALL_XML = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# Tags look like <|DSML|invoke ...> or <｜DSML｜invoke ...> (fullwidth bars).
_DSML_INVOKE = re.compile(
    r"<(?:\|+|｜+)DSML(?:\|+|｜+)\s*invoke\s+name=\"([^\"]+)\">(.*?)</(?:\|+|｜+)DSML(?:\|+|｜+)\s*invoke>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM = re.compile(
    r"<(?:\|+|｜+)DSML(?:\|+|｜+)\s*parameter\s+name=\"([^\"]+)\"([^>]*)>(.*?)</(?:\|+|｜+)DSML(?:\|+|｜+)\s*parameter>",
    re.DOTALL | re.IGNORECASE,
)


def render_tools(schemas: list[dict]) -> str:
    """Describe tools and the call format for the system prompt."""
    lines = [
        "You can call tools using this exact format and no other:",
        "",
        BEGIN,
        '{"name": "<tool_name>", "args": {<json object of arguments>}}',
        END,
        "",
        "Rules:",
        "- Put any natural-language reply text ABOVE the tool-call block.",
        "- Use at most ONE tool call per reply.",
        "- args must be valid JSON and keep JSON types (1, 1.0, true, \"1\" differ).",
        "- If you can answer without a tool, reply with plain text and no tool-call block.",
        "- Never use XML, <tool_call>, DSML, function_call, or any other markup for tools.",
        "- Never invent a different tool-call syntax. Only the @@@ markers above are valid.",
        "",
        "Available tools:",
    ]
    for schema in schemas:
        name = schema.get("name", "")
        desc = schema.get("description", "")
        params = schema.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        props = (params or {}).get("properties") or {}
        required = (params or {}).get("required") or []
        if props:
            parts = []
            for key, meta in props.items():
                typ = meta.get("type", "any") if isinstance(meta, dict) else "any"
                req = "required" if key in required else "optional"
                parts.append(f"{key} ({typ}, {req})")
            lines.append("  parameters: " + ", ".join(parts))
        else:
            lines.append("  parameters: (none)")
        # Include raw schema so the model sees enums/constraints.
        lines.append("  schema: " + json.dumps(params, ensure_ascii=False))
    return "\n".join(lines)


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False)
    block = f"{BEGIN}\n{payload}\n{END}"
    if text:
        return f"{text}\n{block}"
    return block


def _as_tool_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("args", obj.get("arguments", {}))
    if not isinstance(name, str) or not name:
        return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def _coerce_param(raw: str, attrs: str) -> Any:
    text = raw.strip()
    is_string = 'string="true"' in attrs.replace(" ", "").lower()
    if is_string:
        return text
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_dsml(text: str) -> ParsedResponse | None:
    invokes = list(_DSML_INVOKE.finditer(text))
    if not invokes:
        return None
    if len(invokes) > 1:
        return ParsedResponse(text="", tool_call=None, error="Multiple tool calls in one reply.")
    m = invokes[0]
    name = m.group(1).strip()
    body = m.group(2)
    args: dict[str, Any] = {}
    for pm in _DSML_PARAM.finditer(body):
        args[pm.group(1)] = _coerce_param(pm.group(3), pm.group(2) or "")
    if not name:
        return ParsedResponse(
            text=text.strip(),
            tool_call=None,
            error="Tool call JSON must be {\"name\": str, \"args\": object}.",
        )
    visible = (text[: m.start()] + text[m.end() :]).strip()
    # Strip wrapping DSML calls tags if present.
    visible = re.sub(
        r"<(?:\|+|｜+)DSML(?:\|+|｜+)\s*calls?>|</(?:\|+|｜+)DSML(?:\|+|｜+)\s*calls?>",
        "",
        visible,
        flags=re.IGNORECASE,
    ).strip()
    return ParsedResponse(text=visible, tool_call=ToolCall(name=name, args=args), error=None)


def _parse_tool_call_xml(text: str) -> ParsedResponse | None:
    matches = list(_TOOL_CALL_XML.finditer(text))
    if not matches:
        return None
    if len(matches) > 1:
        return ParsedResponse(text="", tool_call=None, error="Multiple tool calls in one reply.")
    m = matches[0]
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ParsedResponse(
            text=text[: m.start()].strip(),
            tool_call=None,
            error="Incomplete or invalid tool call JSON.",
        )
    call = _as_tool_call(obj)
    if call is None:
        return ParsedResponse(
            text=(text[: m.start()] + text[m.end() :]).strip(),
            tool_call=None,
            error="Tool call JSON must be {\"name\": str, \"args\": object}.",
        )
    visible = (text[: m.start()] + text[m.end() :]).strip()
    return ParsedResponse(text=visible, tool_call=call, error=None)


def _parse_course_format(text: str) -> ParsedResponse:
    start = text.find(BEGIN)
    if start < 0:
        return ParsedResponse(text=text.strip(), tool_call=None, error=None)

    after = text[start + len(BEGIN) :]
    if after.startswith("\n"):
        after = after[1:]

    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(after)
    except json.JSONDecodeError:
        return ParsedResponse(
            text=text[:start].strip(),
            tool_call=None,
            error="Incomplete or invalid tool call JSON.",
        )

    rest = after[idx:]
    if rest.startswith("\n"):
        rest = rest[1:]
    if not rest.startswith(END):
        return ParsedResponse(
            text=text[:start].strip(),
            tool_call=None,
            error="Tool call is truncated or missing the end marker.",
        )

    after_end = rest[len(END) :]
    if BEGIN in after_end:
        return ParsedResponse(
            text=text[:start].strip(),
            tool_call=None,
            error="Multiple tool calls in one reply.",
        )

    call = _as_tool_call(obj)
    if call is None:
        return ParsedResponse(
            text=(text[:start] + after_end).strip(),
            tool_call=None,
            error="Tool call JSON must be {\"name\": str, \"args\": object}.",
        )

    visible = (text[:start] + after_end).strip()
    return ParsedResponse(text=visible, tool_call=call, error=None)


def parse_response(text: str) -> ParsedResponse:
    """Never raises. At most one tool call; truncated/invalid calls set error."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    if BEGIN in text:
        return _parse_course_format(text)

    # Fallbacks for models that ignore the course markers.
    alt = _parse_tool_call_xml(text)
    if alt is not None:
        return alt
    alt = _parse_dsml(text)
    if alt is not None:
        return alt

    return ParsedResponse(text=text.strip(), tool_call=None, error=None)
