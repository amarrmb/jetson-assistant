import pytest
from jetson_assistant.backends.tool_parser import ToolParser


class FakeToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def test_plain_text_no_tools():
    """Text without brackets should pass through unchanged."""
    texts = []
    tools = []
    parser = ToolParser(on_text=texts.append, on_tool=tools.append)
    for token in ["Hello", " world", "!"]:
        parser.feed(token)
    parser.flush()
    assert "".join(texts) == "Hello world!"
    assert tools == []


def test_single_bracket_tool():
    """[look left] should be extracted as a tool call."""
    texts = []
    tools = []
    parser = ToolParser(on_text=texts.append, on_tool=tools.append)
    for token in ["Sure", " ", "[", "look", " left", "]", " Let me check"]:
        parser.feed(token)
    parser.flush()
    assert "look left" in [t["raw"] for t in tools]
    assert "[" not in "".join(texts)
    assert "]" not in "".join(texts)


def test_tool_parsing_look():
    """[look left] should map to tool name='look', args={'direction': 'left'}."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[look left]")
    parser.flush()
    assert len(tools) == 1
    assert tools[0]["name"] == "look"
    assert tools[0]["args"] == {"direction": "left"}


def test_tool_parsing_express():
    """[express happy] should map to tool name='express', args={'emotion': 'happy'}."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[express happy]")
    parser.flush()
    assert tools[0]["name"] == "express"
    assert tools[0]["args"] == {"emotion": "happy"}


def test_tool_parsing_dance():
    """[dance] with no arg should default to name='random'."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[dance]")
    parser.flush()
    assert tools[0]["name"] == "dance"
    assert tools[0]["args"] == {"name": "random"}


def test_tool_parsing_nod():
    """[nod yes] should map correctly."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[nod yes]")
    parser.flush()
    assert tools[0]["name"] == "nod"
    assert tools[0]["args"] == {"response": "yes"}


def test_multiple_tools_in_stream():
    """Multiple tools in one stream should all be extracted."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    for token in ["[look left]", " Oh ", "[express curious]", " interesting"]:
        parser.feed(token)
    parser.flush()
    assert len(tools) == 2
    assert tools[0]["name"] == "look"
    assert tools[1]["name"] == "express"


def test_incomplete_bracket_flushed_as_text():
    """Unclosed bracket should flush as plain text."""
    texts = []
    tools = []
    parser = ToolParser(on_text=texts.append, on_tool=tools.append)
    parser.feed("[oops no closing")
    parser.flush()
    assert tools == []
    assert "[oops no closing" in "".join(texts)


def test_unknown_tool_ignored():
    """[unknown command] should pass through as text."""
    texts = []
    tools = []
    parser = ToolParser(on_text=texts.append, on_tool=tools.append)
    parser.feed("[fly away]")
    parser.flush()
    assert tools == []
    assert "[fly away]" in "".join(texts)


def test_tool_parsing_search():
    """[search who won the superbowl] → web_search(query='who won the superbowl')."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[search who won the superbowl]")
    parser.flush()
    assert len(tools) == 1
    assert tools[0]["name"] == "web_search"
    assert tools[0]["args"] == {"query": "who won the superbowl"}


def test_tool_parsing_web_search():
    """[web_search latest AI news] → web_search(query='latest AI news')."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[web_search latest AI news]")
    parser.flush()
    assert len(tools) == 1
    assert tools[0]["name"] == "web_search"
    assert tools[0]["args"] == {"query": "latest AI news"}


def test_tool_parsing_search_default():
    """[search] with no arg should default to 'latest news'."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[search]")
    parser.flush()
    assert tools[0]["name"] == "web_search"
    assert tools[0]["args"] == {"query": "latest news"}


def test_tool_parsing_camera():
    """[camera what do you see] → check_camera(question='what do you see')."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[camera what do you see]")
    parser.flush()
    assert len(tools) == 1
    assert tools[0]["name"] == "check_camera"
    assert tools[0]["args"] == {"question": "what do you see"}


def test_tool_parsing_see():
    """[see what's on the table] → check_camera(question="what's on the table")."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[see what's on the table]")
    parser.flush()
    assert len(tools) == 1
    assert tools[0]["name"] == "check_camera"
    assert tools[0]["args"] == {"question": "what's on the table"}


def test_tool_parsing_describe():
    """[describe] with no arg should default to 'Describe what you see'."""
    tools = []
    parser = ToolParser(on_text=lambda t: None, on_tool=tools.append)
    parser.feed("[describe]")
    parser.flush()
    assert tools[0]["name"] == "check_camera"
    assert tools[0]["args"] == {"question": "Describe what you see"}
