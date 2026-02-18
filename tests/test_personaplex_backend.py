import pytest
import time
from unittest.mock import patch, MagicMock
from jetson_assistant.assistant.core import AssistantConfig


def test_personaplex_config_defaults():
    """PersonaPlex config should have sensible defaults."""
    config = AssistantConfig(mode="personaplex")
    assert config.personaplex_audio == "browser"
    assert config.personaplex_port == 8998
    assert config.personaplex_voice == "NATF1"
    assert config.personaplex_tool_detection == "prompt"


def test_personaplex_backend_init():
    """PersonaplexBackend should initialize without GPU (lazy loading)."""
    from jetson_assistant.backends.personaplex import PersonaplexBackend

    config = AssistantConfig(mode="personaplex")
    # Should not load model at init — lazy loading
    backend = PersonaplexBackend(config)
    assert backend.config.mode == "personaplex"
    assert not backend.is_loaded()


# ── Tool registration tests ──

def test_personaplex_tools_get_time_and_web_search():
    """PersonaPlex mode should register get_time and web_search (no vision)."""
    config = AssistantConfig(mode="personaplex", vision_enabled=False)
    from jetson_assistant.assistant.core import VoiceAssistant
    assistant = VoiceAssistant(config=config)
    tools = assistant._tools

    # get_time should be registered
    assert "get_time" in tools._tools
    result = tools._tools["get_time"]["fn"]()
    assert ":" in result  # contains time

    # web_search should be registered
    assert "web_search" in tools._tools


def test_personaplex_tools_check_camera_with_mock():
    """PersonaPlex mode registers check_camera when vision_enabled + camera opens."""
    config = AssistantConfig(
        mode="personaplex",
        vision_enabled=True,
        camera_device=99,
        llm_backend="simple",
        llm_host="http://localhost:11434",
    )

    mock_cam = MagicMock()
    mock_cam.open.return_value = True
    mock_cam.capture_base64.return_value = "fake_base64_image"

    # Patch at the source module where _init_tools_personaplex imports from
    with patch("jetson_assistant.assistant.vision.Camera", return_value=mock_cam), \
         patch("jetson_assistant.assistant.vision.CameraConfig"):
        from jetson_assistant.assistant.core import VoiceAssistant
        assistant = VoiceAssistant(config=config)
        tools = assistant._tools

        assert "check_camera" in tools._tools
        assert hasattr(assistant, "_personaplex_camera")
        assert assistant._personaplex_camera is mock_cam


def test_personaplex_tools_no_camera_when_vision_disabled():
    """PersonaPlex mode should NOT register check_camera when vision_enabled=False."""
    config = AssistantConfig(mode="personaplex", vision_enabled=False)
    from jetson_assistant.assistant.core import VoiceAssistant
    assistant = VoiceAssistant(config=config)
    assert "check_camera" not in assistant._tools._tools


# ── Transcript state tests ──

def test_transcript_stream():
    """Transcript stream entries should be appended with IDs."""
    from jetson_assistant.backends.personaplex import PersonaplexBackend
    config = AssistantConfig(mode="personaplex")
    backend = PersonaplexBackend(config)

    backend.add_transcript_stream("Hello")
    backend.add_transcript_stream(" world")

    assert len(backend._transcript) == 2
    assert backend._transcript[0]["type"] == "stream"
    assert backend._transcript[0]["text"] == "Hello"
    assert backend._transcript[1]["text"] == " world"
    assert backend._transcript[0]["id"] < backend._transcript[1]["id"]


def test_transcript_tool_call_and_result():
    """Tool call + result should create linked transcript entries."""
    from jetson_assistant.backends.personaplex import PersonaplexBackend
    config = AssistantConfig(mode="personaplex")
    backend = PersonaplexBackend(config)

    tool_id = backend.add_tool_call("web_search", {"query": "test"})
    assert tool_id > 0

    # Find the tool entry
    tool_entry = [e for e in backend._transcript if e["type"] == "tool"][0]
    assert tool_entry["name"] == "web_search"
    assert tool_entry["args"] == {"query": "test"}
    assert tool_entry["tool_id"] == tool_id

    # Add result
    backend.add_tool_result(tool_id, "Search result here")
    result_entry = [e for e in backend._transcript if e["type"] == "tool_result"][0]
    assert result_entry["tool_id"] == tool_id
    assert result_entry["result"] == "Search result here"


def test_transcript_html_template():
    """Transcript HTML should contain expected elements."""
    from jetson_assistant.backends.personaplex import _PERSONAPLEX_TRANSCRIPT_HTML
    # Template uses %% for literal percent signs and %(key)s for substitution
    html = _PERSONAPLEX_TRANSCRIPT_HTML % {"camera_class": "", "chat_class": ""}
    assert "/events" in html
    assert "PersonaPlex" in html
    assert "tool-result" in html
    assert "EventSource" in html
    assert "FULL-DUPLEX" in html


def test_transcript_cap_at_200():
    """Transcript should cap at 200 entries."""
    from jetson_assistant.backends.personaplex import PersonaplexBackend
    config = AssistantConfig(mode="personaplex")
    backend = PersonaplexBackend(config)

    for i in range(250):
        backend.add_transcript_stream(f"token_{i}")

    assert len(backend._transcript) == 200
    # Should contain the most recent entries
    assert backend._transcript[-1]["text"] == "token_249"


def test_async_tool_execution():
    """Tool execution in thread pool should not block and should deliver result."""
    from jetson_assistant.backends.personaplex import PersonaplexBackend
    from jetson_assistant.assistant.tools import ToolRegistry
    from typing import Annotated

    config = AssistantConfig(mode="personaplex")
    backend = PersonaplexBackend(config)

    # Create a mock tool registry with a slow tool
    registry = ToolRegistry()

    @registry.register("A test tool")
    def test_tool(query: Annotated[str, "test"] = "default") -> str:
        time.sleep(0.1)
        return f"Result for: {query}"

    backend.set_callbacks(tool_registry=registry)

    # Simulate tool detection via the on_tool_detected pattern from run_browser
    tool_id = backend.add_tool_call("test_tool", {"query": "hello"})

    def _run_tool():
        class _TC:
            def __init__(self, n, a):
                self.name = n
                self.arguments = a
        result = registry.execute(_TC("test_tool", {"query": "hello"}))
        backend.add_tool_result(tool_id, str(result))

    backend._tool_executor.submit(_run_tool)

    # Wait for thread to complete
    time.sleep(0.3)

    # Verify result was added
    result_entries = [e for e in backend._transcript if e["type"] == "tool_result"]
    assert len(result_entries) == 1
    assert "Result for: hello" in result_entries[0]["result"]
