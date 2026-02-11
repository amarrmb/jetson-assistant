"""Tests for config loading, merge logic, and RTSP sanitization."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from jetson_speech.assistant.core import AssistantConfig

# ---------------------------------------------------------------------------
# Helpers — mirrors the logic in cli.py and core.py
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
PRESET_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))


def _sanitize_url(url: str) -> str:
    """Mirrors VoiceAssistant._sanitize_url (static method)."""
    return re.sub(r"(rtsp://)([^@]+)@", r"\1***@", url)


def _resolve(cli_name, cli_overrides, yaml_config, cli_locals, config_name=None):
    """Mirrors the _resolve() helper in cli.py's assistant command."""
    cfg_key = config_name or cli_name
    if cli_name in cli_overrides:
        return cli_overrides[cli_name]
    if cfg_key in yaml_config:
        return yaml_config[cfg_key]
    return cli_locals[cli_name]


# ---------------------------------------------------------------------------
# from_yaml() — loads presets
# ---------------------------------------------------------------------------


class TestFromYaml:
    @pytest.mark.parametrize("preset", PRESET_FILES, ids=lambda p: p.stem)
    def test_loads_all_presets(self, preset):
        """Each preset loads without error and returns a dict with known keys."""
        result = AssistantConfig.from_yaml(str(preset))
        assert isinstance(result, dict)
        assert len(result) > 0
        valid_keys = {f.name for f in fields(AssistantConfig)}
        for key in result:
            assert key in valid_keys, f"Unexpected key '{key}' leaked from {preset.name}"

    @pytest.mark.parametrize("preset", PRESET_FILES, ids=lambda p: p.stem)
    def test_preset_produces_valid_config(self, preset):
        """AssistantConfig(**from_yaml(path)) must not crash."""
        data = AssistantConfig.from_yaml(str(preset))
        config = AssistantConfig(**data)
        assert config.llm_backend in ("ollama", "vllm", "openai", "anthropic", "simple")

    def test_filters_unknown_keys(self, tmp_path):
        """YAML with bogus_key should not bleed into returned dict."""
        p = tmp_path / "bad.yaml"
        p.write_text("bogus_key: 123\nllm_backend: vllm\n")
        result = AssistantConfig.from_yaml(str(p))
        assert "bogus_key" not in result
        assert result["llm_backend"] == "vllm"

    def test_excludes_callbacks(self, tmp_path):
        """Callback fields (on_wake etc.) must be filtered out."""
        p = tmp_path / "cb.yaml"
        p.write_text("on_wake: foo\non_error: bar\nverbose: true\n")
        result = AssistantConfig.from_yaml(str(p))
        assert "on_wake" not in result
        assert "on_error" not in result
        assert result.get("verbose") is True

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file returns {}."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = AssistantConfig.from_yaml(str(p))
        assert result == {}

    def test_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AssistantConfig.from_yaml("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# RTSP sanitization
# ---------------------------------------------------------------------------


class TestSanitizeUrl:
    def test_strips_credentials(self):
        url = "rtsp://admin:secret@192.0.2.50:554/stream1"
        assert _sanitize_url(url) == "rtsp://***@192.0.2.50:554/stream1"

    def test_strips_user_only(self):
        url = "rtsp://user@192.0.2.50:554/stream1"
        assert _sanitize_url(url) == "rtsp://***@192.0.2.50:554/stream1"

    def test_non_rtsp_passes_through(self):
        url = "http://localhost:8001/v1/models"
        assert _sanitize_url(url) == url

    def test_no_credentials_passes_through(self):
        url = "rtsp://192.0.2.50:554/stream1"
        assert _sanitize_url(url) == url


# ---------------------------------------------------------------------------
# CLI merge logic (_resolve pattern)
# ---------------------------------------------------------------------------


class TestResolveLogic:
    """Test the CLI > YAML > default merge priority."""

    def test_cli_override_beats_yaml(self):
        cli_overrides = {"llm_backend": "openai"}
        yaml_config = {"llm_backend": "vllm"}
        cli_locals = {"llm_backend": "ollama"}
        assert _resolve("llm_backend", cli_overrides, yaml_config, cli_locals) == "openai"

    def test_yaml_beats_default(self):
        cli_overrides = {}
        yaml_config = {"llm_backend": "vllm"}
        cli_locals = {"llm_backend": "ollama"}
        assert _resolve("llm_backend", cli_overrides, yaml_config, cli_locals) == "vllm"

    def test_default_when_no_override(self):
        cli_overrides = {}
        yaml_config = {}
        cli_locals = {"llm_backend": "ollama"}
        assert _resolve("llm_backend", cli_overrides, yaml_config, cli_locals) == "ollama"

    def test_config_name_mapping(self):
        """YAML key can differ from CLI key (e.g., 'voice' → 'tts_voice')."""
        cli_overrides = {}
        yaml_config = {"tts_voice": "af_heart"}
        cli_locals = {"voice": "en_US-amy-medium"}
        assert (
            _resolve("voice", cli_overrides, yaml_config, cli_locals, config_name="tts_voice")
            == "af_heart"
        )
