"""Parse PersonaPlex text token stream for [tool commands]."""

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Known tools and their argument mapping
# Format: tool_keyword -> (tool_name, arg_name, default_value)
_TOOL_MAP = {
    "look": ("look", "direction", "center"),
    "express": ("express", "emotion", "happy"),
    "dance": ("dance", "name", "random"),
    "nod": ("nod", "response", "yes"),
    "reachy_see": ("reachy_see", "question", "Describe what you see"),
    "reachy_power": ("reachy_power", "action", "wake"),
    "set_antennas": None,  # special handling
    "look_at_point": None,  # special handling
    "reachy_status": ("reachy_status", None, None),
    # Web search
    "search": ("web_search", "query", "latest news"),
    "web_search": ("web_search", "query", "latest news"),
    # Camera vision
    "camera": ("check_camera", "question", "Describe what you see"),
    "see": ("check_camera", "question", "Describe what you see"),
    "describe": ("check_camera", "question", "Describe what you see"),
}


class ToolParser:
    """Parse streaming text tokens for [tool command] brackets.

    Calls on_text(str) for plain text and on_tool(dict) for tool triggers.
    Tool dict: {"name": str, "args": dict, "raw": str}
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_tool: Callable[[dict], None],
    ):
        self._on_text = on_text
        self._on_tool = on_tool
        self._buffer = ""
        self._in_bracket = False
        self._bracket_buf = ""

    def feed(self, token: str) -> None:
        """Feed a text token from PersonaPlex's output stream."""
        for ch in token:
            if ch == "[" and not self._in_bracket:
                # Flush any buffered plain text
                if self._buffer:
                    self._on_text(self._buffer)
                    self._buffer = ""
                self._in_bracket = True
                self._bracket_buf = ""
            elif ch == "]" and self._in_bracket:
                self._in_bracket = False
                self._dispatch_bracket(self._bracket_buf.strip())
                self._bracket_buf = ""
            elif self._in_bracket:
                self._bracket_buf += ch
            else:
                self._buffer += ch

    def flush(self) -> None:
        """Flush remaining buffer (call at end of stream or sentence boundary)."""
        if self._in_bracket and self._bracket_buf:
            # Unclosed bracket — emit as plain text
            self._on_text("[" + self._bracket_buf)
            self._in_bracket = False
            self._bracket_buf = ""
        if self._buffer:
            self._on_text(self._buffer)
            self._buffer = ""

    def _dispatch_bracket(self, raw: str) -> None:
        """Parse a bracket command and dispatch as tool call."""
        parts = raw.split(None, 1)
        if not parts:
            return

        keyword = parts[0].lower()
        arg_str = parts[1].strip() if len(parts) > 1 else None

        mapping = _TOOL_MAP.get(keyword)
        if mapping is None and keyword not in _TOOL_MAP:
            # Unknown tool — emit as plain text
            self._on_text(f"[{raw}]")
            return

        if mapping is None:
            # Special handling tools (set_antennas, look_at_point) — skip for now
            self._on_text(f"[{raw}]")
            return

        tool_name, arg_name, default = mapping
        args = {}
        if arg_name is not None:
            args[arg_name] = arg_str if arg_str else default

        self._on_tool({"name": tool_name, "args": args, "raw": raw})
        logger.debug("Tool detected: %s(%s)", tool_name, args)
