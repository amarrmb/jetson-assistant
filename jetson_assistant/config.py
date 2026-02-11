"""
Configuration and settings for Jetson Assistant.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def get_default_cache_dir() -> Path:
    """Get the default model cache directory."""
    return Path(os.environ.get("JETSON_ASSISTANT_MODEL_CACHE", Path.home() / ".cache" / "jetson-assistant"))


def is_jetson() -> bool:
    """Check if running on a Jetson device."""
    # Check device tree
    device_tree = Path("/proc/device-tree/compatible")
    if device_tree.exists():
        content = device_tree.read_text(errors="ignore").lower()
        if "tegra" in content or "nvidia" in content:
            return True

    # Check hostname
    hostname = os.uname().nodename.lower()
    if "jetson" in hostname:
        return True

    return False


def get_jetson_power_mode() -> str | None:
    """Get the current Jetson power mode."""
    nvpmodel = Path("/usr/bin/nvpmodel")
    if not nvpmodel.exists():
        return None

    try:
        import subprocess
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse output like "NV Power Mode: MAXN"
        for line in result.stdout.splitlines():
            if "NV Power Mode:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass

    return None


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(
        default_factory=lambda: os.environ.get("JETSON_ASSISTANT_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.environ.get("JETSON_ASSISTANT_PORT", "8080"))
    )
    cors_origins: list[str] = Field(default=["*"])
    enable_webui: bool = Field(default=False)


class TTSConfig(BaseModel):
    """TTS configuration."""

    default_backend: str = Field(
        default_factory=lambda: os.environ.get("JETSON_ASSISTANT_TTS_BACKEND", "qwen")
    )
    default_voice: str = Field(default="ryan")
    default_language: str = Field(default="English")
    chunk_size: int = Field(default=500)  # Max chars per chunk


class STTConfig(BaseModel):
    """STT configuration."""

    default_backend: str = Field(
        default_factory=lambda: os.environ.get("JETSON_ASSISTANT_STT_BACKEND", "whisper")
    )
    default_model_size: str = Field(default="base")
    default_language: str | None = Field(default=None)  # Auto-detect


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    iterations: int = Field(default=3)
    warmup_iterations: int = Field(default=1)
    output_format: Literal["json", "markdown", "html"] = Field(default="markdown")


class Config(BaseModel):
    """Main configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    model_cache_dir: Path = Field(default_factory=get_default_cache_dir)
    is_jetson: bool = Field(default_factory=is_jetson)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _config
    _config = config
