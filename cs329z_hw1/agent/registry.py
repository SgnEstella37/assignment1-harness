"""Problem 5 (tool_registry): validate and execute tools safely."""

from __future__ import annotations

from typing import Any, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools: list[ToolSpec] = list(tools)
        self._by_name: dict[str, ToolSpec] = {t.name: t for t in self._tools}
        self._validators: dict[str, Draft202012Validator] = {
            t.name: Draft202012Validator(t.parameters) for t in self._tools
        }

    def schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools
        ]

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._by_name.get(name)

    def validate(self, name: str, args: dict) -> ToolResult | None:
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult(status="unknown_tool", content=f"Unknown tool: {name!r}")
        try:
            self._validators[name].validate(args if isinstance(args, dict) else {})
        except ValidationError as exc:
            return ToolResult(
                status="invalid_args",
                content=f"Invalid arguments for {name}: {exc.message}",
            )
        except Exception as exc:  # malformed schema / args type
            return ToolResult(
                status="invalid_args",
                content=f"Invalid arguments for {name}: {exc}",
            )
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        checked = self.validate(name, args)
        if checked is not None:
            return checked
        tool = self._by_name[name]
        try:
            value = tool.fn(**args)
        except Exception as exc:
            return ToolResult(status="tool_error", content=f"Tool {name} failed: {exc}")
        return ToolResult(status="ok", content=str(value))


def build_registry(tools: list[ToolSpec]) -> ToolRegistry:
    return ToolRegistry(tools)
