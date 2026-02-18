import pytest
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
