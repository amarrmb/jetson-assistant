"""Parse PersonaPlex text token stream for [tool commands] and keyword triggers."""

import logging
import re
import time
from typing import Callable, Optional

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


# ── Keyword-based tool detection for PersonaPlex ──
#
# Moshi 7B is a speech-to-speech model that doesn't reliably emit structured
# [bracket] commands from system prompts. KeywordToolDetector monitors the
# model's text output and triggers tools based on keyword patterns.
#
# Pattern categories:
#   SEARCH — factual questions the model is likely to hallucinate about
#   CAMERA — requests to look at / describe the visual scene
#   TIME   — time queries (trivial, no hallucination risk)

# High-priority patterns — checked on EVERY token (fire immediately, don't wait for sentence end)
_URGENT_SEARCH_PATTERNS = [
    # Specific factual topics — model is about to hallucinate
    (re.compile(r"\b(?:super ?bowl|world cup|election|oscars?|grammy|nobel)\b", re.I), None),
    # Question patterns (model echoing user's question)
    (re.compile(r"\bwho (?:won|is|was|are|were)\b", re.I), None),
    (re.compile(r"\bwhat (?:is|are|was|were) (?:the |a )?(?:latest|score|result|news|price|weather)\b", re.I), None),
]

# Normal-priority patterns — checked at sentence boundaries
_SEARCH_PATTERNS = [
    (re.compile(r"\bwhen (?:did|is|was|will)\b", re.I), None),
    (re.compile(r"\bhow (?:much|many|old|tall|far)\b", re.I), None),
    # Explicit search requests paraphrased by model
    (re.compile(r"\bsearch(?:ing)? for\b", re.I), None),
    (re.compile(r"\b(?:latest|recent|current) (?:news|updates|score|results)\b", re.I), None),
]

_CAMERA_PATTERNS = [
    (re.compile(r"\b(?:i can see|let me (?:look|see)|looking at|i see)\b", re.I), None),
    (re.compile(r"\b(?:what do you see|what can you see|describe what|show me)\b", re.I), None),
    (re.compile(r"\b(?:in front of|through the lens)\b", re.I), None),
    # Hardware/device questions about cameras
    (re.compile(r"\b(?:how many|number of) (?:cameras?|sensors?)\b", re.I), None),
    (re.compile(r"\bcameras? (?:on ?board|do (?:you|we|i) have|are there|installed)\b", re.I), None),
]

_TIME_PATTERNS = [
    (re.compile(r"\b(?:current time|what time|the time is)\b", re.I), None),
]


class KeywordToolDetector:
    """Detect tool-triggering keywords in PersonaPlex model text output.

    Monitors the streaming text, accumulates into sentences, and fires
    tool callbacks when keyword patterns match. Uses cooldowns to prevent
    repeated triggers.

    Args:
        on_tool: Callback with (tool_name, args_dict) when a tool should fire.
        cooldown: Minimum seconds between firing the same tool.
        registered_tools: Set of tool names that are actually available.
    """

    def __init__(
        self,
        on_tool: Callable[[str, dict], None],
        cooldown: float = 15.0,
        registered_tools: Optional[set] = None,
    ):
        self._on_tool = on_tool
        self._cooldown = cooldown
        self._registered = registered_tools or {"web_search", "check_camera", "get_time"}
        self._sentence_buf = ""
        self._last_fire: dict[str, float] = {}
        self._token_count = 0

    def feed(self, token: str) -> None:
        """Feed a text token from the model's output stream."""
        self._sentence_buf += token
        self._token_count += 1

        # Check urgent patterns on EVERY token (fire before model hallucinates)
        self._check_urgent()

        # Check all patterns at sentence boundaries or every ~30 tokens
        if self._at_sentence_end(token) or self._token_count >= 30:
            self._check_and_fire()
            # Keep a sliding window — trim old text but keep recent context
            if len(self._sentence_buf) > 200:
                self._sentence_buf = self._sentence_buf[-100:]
            self._token_count = 0

    def flush(self) -> None:
        """Check remaining buffer at end of stream."""
        if self._sentence_buf.strip():
            self._check_and_fire()
        self._sentence_buf = ""
        self._token_count = 0

    def _at_sentence_end(self, token: str) -> bool:
        return any(ch in token for ch in ".!?")

    def _can_fire(self, tool_name: str) -> bool:
        last = self._last_fire.get(tool_name, 0.0)
        return (time.time() - last) >= self._cooldown

    def _fire(self, tool_name: str, args: dict) -> None:
        if not self._can_fire(tool_name):
            return
        if tool_name not in self._registered:
            return
        self._last_fire[tool_name] = time.time()
        logger.info("Keyword trigger: %s(%s) from: ...%s", tool_name, args, self._sentence_buf[-60:])
        self._on_tool(tool_name, args)

    def _check_urgent(self) -> None:
        """Check high-priority patterns on every token (don't wait for sentence end)."""
        text = self._sentence_buf.lower()
        if len(text) < 6:
            return
        for pattern, _ in _URGENT_SEARCH_PATTERNS:
            m = pattern.search(text)
            if m:
                query = self._extract_search_query(m)
                self._fire("web_search", {"query": query})
                self._sentence_buf = ""
                self._token_count = 0
                return

    def _extract_search_query(self, match: re.Match) -> str:
        """Extract a clean search query from the matched pattern.

        For topic patterns (super bowl, election, etc.), construct a
        direct question query instead of using the model's text (which
        may contain hallucinated answers).
        """
        matched = match.group().lower().strip()

        # Topic-specific clean queries — don't use model's fabricated answer
        topic_queries = {
            "super bowl": "who won the Super Bowl 2026",
            "superbowl": "who won the Super Bowl 2026",
            "world cup": "who won the World Cup latest results",
            "election": "latest election results 2026",
            "oscars": "Oscar winners 2026",
            "oscar": "Oscar winners 2026",
            "grammy": "Grammy winners 2026",
            "nobel": "Nobel Prize winners latest",
        }
        for topic, query in topic_queries.items():
            if topic in matched:
                return query

        # For question patterns (who won, what is the latest, etc.),
        # use the text around the match but truncate at the answer
        text = self._sentence_buf.strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        match_text = match.group()
        for sent in reversed(sentences):
            if match_text.lower() in sent.lower():
                query = sent.strip()
                break
        else:
            query = text[-60:].strip()

        # Remove filler prefixes
        for prefix in ("well ", "so ", "let me ", "i think ", "actually ", "oh ", "hmm ",
                        "alright ", "okay ", "sure "):
            if query.lower().startswith(prefix):
                query = query[len(prefix):]

        # Truncate at answer indicators — keep the question, drop the model's answer
        for splitter in (" was won by", " is won by", " the winner is", " the answer is",
                         " the result is", " it was ", " it is "):
            idx = query.lower().find(splitter)
            if idx > 10:  # only if there's enough question before the split
                query = query[:idx]
                break

        if len(query) > 80:
            query = query[:80]
        return query.strip() or "latest news"

    def _check_and_fire(self) -> None:
        text = self._sentence_buf.lower()
        if len(text) < 8:
            return

        # Check camera patterns FIRST (more specific — "how many cameras"
        # would otherwise match generic "how many" search pattern)
        for pattern, _ in _CAMERA_PATTERNS:
            if pattern.search(text):
                self._fire("check_camera", {"question": "Describe what you see in detail"})
                self._sentence_buf = ""
                return

        # Check time patterns (trivial, no hallucination risk)
        for pattern, _ in _TIME_PATTERNS:
            if pattern.search(text):
                self._fire("get_time", {})
                self._sentence_buf = ""
                return

        # Check urgent search patterns (fallback — may have been missed if buffer was short)
        for pattern, _ in _URGENT_SEARCH_PATTERNS:
            m = pattern.search(text)
            if m:
                query = self._extract_search_query(m)
                self._fire("web_search", {"query": query})
                self._sentence_buf = ""
                return

        # Check normal search patterns
        for pattern, _ in _SEARCH_PATTERNS:
            m = pattern.search(text)
            if m:
                query = self._extract_search_query(m)
                self._fire("web_search", {"query": query})
                self._sentence_buf = ""
                return
