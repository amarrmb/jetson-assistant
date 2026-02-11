"""
ToolRegistry — auto-generates OpenAI-format tool schemas from Python type hints.

Replaces hand-written JSON tool definitions and if/elif dispatchers.
"""

import inspect
import logging
import re
from typing import Annotated, Callable, Optional, get_args, get_origin

logger = logging.getLogger(__name__)


# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _parse_google_docstring_args(fn: Callable) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstring Args: section."""
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    descriptions: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            # End of Args section on blank line or new section header
            if not stripped or (stripped.endswith(":") and not stripped.startswith(" ")):
                if not stripped.startswith(" ") and ":" in stripped and stripped[0].isalpha():
                    break
                if not stripped:
                    break
            # Match "param_name: description" or "param_name (type): description"
            m = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
            if m:
                descriptions[m.group(1)] = m.group(2).strip()
    return descriptions


class ToolRegistry:
    """Registry that turns decorated Python functions into OpenAI tool schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}  # name → {"fn": callable, "schema": dict}

    def register(self, description: str) -> Callable:
        """Decorator that registers a function as an LLM-callable tool.

        Inspects the function's type hints to build an OpenAI-format JSON Schema.
        Parameter descriptions come from ``Annotated[type, "desc"]`` (priority)
        or from Google-style docstring ``Args:`` section.

        Args:
            description: Human-readable description of what the tool does.
        """
        def decorator(fn: Callable) -> Callable:
            sig = inspect.signature(fn)
            hints = fn.__annotations__ if hasattr(fn, '__annotations__') else {}
            docstring_args = _parse_google_docstring_args(fn)

            properties: dict = {}
            required: list[str] = []

            for name, param in sig.parameters.items():
                hint = hints.get(name)
                if hint is None:
                    continue

                # Resolve Annotated[T, "desc"]
                param_desc: Optional[str] = None
                actual_type = hint
                if get_origin(hint) is Annotated:
                    args = get_args(hint)
                    actual_type = args[0]
                    # Look for a string annotation as the description
                    for a in args[1:]:
                        if isinstance(a, str):
                            param_desc = a
                            break

                # Fall back to docstring description
                if param_desc is None:
                    param_desc = docstring_args.get(name)

                json_type = _TYPE_MAP.get(actual_type, "string")
                prop: dict = {"type": json_type}
                if param_desc:
                    prop["description"] = param_desc
                properties[name] = prop

                # Required if no default value
                if param.default is inspect.Parameter.empty:
                    required.append(name)

            schema: dict = {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
            if required:
                schema["function"]["parameters"]["required"] = required

            self._tools[fn.__name__] = {"fn": fn, "schema": schema}
            return fn

        return decorator

    def definitions(self) -> list[dict]:
        """Return OpenAI-format tool list for the LLM."""
        return [entry["schema"] for entry in self._tools.values()]

    def execute(self, tool_call) -> Optional[str]:
        """Route and execute a tool call.

        Args:
            tool_call: Object with `name` (str) and `arguments` (dict) attributes.

        Returns:
            Return value from the handler (stringified), or None if unknown tool.
        """
        entry = self._tools.get(tool_call.name)
        if entry is None:
            return None

        fn = entry["fn"]
        args = tool_call.arguments if hasattr(tool_call, "arguments") else {}

        try:
            result = fn(**args)
            logger.debug("Tool: %s(%s)", tool_call.name, args)
            return str(result) if result is not None else None
        except Exception as e:
            logger.error("Tool error: %s: %s", tool_call.name, e)
            return None

    def __bool__(self) -> bool:
        return len(self._tools) > 0

    def __len__(self) -> int:
        return len(self._tools)
