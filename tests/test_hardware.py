"""Tests for hardware detection and tier auto-selection."""

from unittest.mock import mock_open, patch

import pytest

from jetson_assistant.hardware import JetsonTier, detect_tier, get_vram_gb


# ---------------------------------------------------------------------------
# JetsonTier enum properties
# ---------------------------------------------------------------------------


class TestJetsonTier:
    def test_tier_values(self):
        assert JetsonTier.THOR.value == "thor"
        assert JetsonTier.ORIN.value == "orin"
        assert JetsonTier.NANO.value == "nano"

    def test_tier_has_config_path(self):
        assert JetsonTier.THOR.config == "configs/thor.yaml"
        assert JetsonTier.ORIN.config == "configs/orin.yaml"
        assert JetsonTier.NANO.config == "configs/nano.yaml"

    def test_tier_has_compose_file(self):
        assert JetsonTier.THOR.compose == "docker-compose.thor.yml"
        assert JetsonTier.ORIN.compose == "docker-compose.orin.yml"
        assert JetsonTier.NANO.compose == "docker-compose.nano.yml"


# ---------------------------------------------------------------------------
# detect_tier() — VRAM-based tier selection
# ---------------------------------------------------------------------------


class TestDetectTier:
    def test_detect_tier_thor(self):
        """>=96GB maps to THOR."""
        assert detect_tier(vram_gb=128) == JetsonTier.THOR
        assert detect_tier(vram_gb=96) == JetsonTier.THOR

    def test_detect_tier_orin(self):
        """>=16GB and <96GB maps to ORIN."""
        assert detect_tier(vram_gb=64) == JetsonTier.ORIN
        assert detect_tier(vram_gb=32) == JetsonTier.ORIN
        assert detect_tier(vram_gb=16) == JetsonTier.ORIN

    def test_detect_tier_nano(self):
        """<16GB maps to NANO."""
        assert detect_tier(vram_gb=8) == JetsonTier.NANO
        assert detect_tier(vram_gb=4) == JetsonTier.NANO

    def test_detect_tier_zero_vram(self):
        """0 VRAM falls to NANO."""
        assert detect_tier(vram_gb=0) == JetsonTier.NANO

    def test_detect_tier_boundary_values(self):
        """Exact boundary values."""
        assert detect_tier(vram_gb=95.9) == JetsonTier.ORIN
        assert detect_tier(vram_gb=96.0) == JetsonTier.THOR
        assert detect_tier(vram_gb=15.9) == JetsonTier.NANO
        assert detect_tier(vram_gb=16.0) == JetsonTier.ORIN

    def test_detect_tier_auto_calls_get_vram(self):
        """When vram_gb is None, detect_tier calls get_vram_gb()."""
        with patch("jetson_assistant.hardware.get_vram_gb", return_value=64.0) as mock:
            tier = detect_tier()
            mock.assert_called_once()
            assert tier == JetsonTier.ORIN


# ---------------------------------------------------------------------------
# get_vram_gb() — GPU/system memory detection
# ---------------------------------------------------------------------------


class TestGetVramGb:
    def test_torch_cuda_path(self):
        """When torch.cuda is available, reads GPU memory."""
        mock_props = type("Props", (), {"total_mem": 128 * (1024**3)})()
        mock_torch = type("MockTorch", (), {
            "cuda": type("Cuda", (), {
                "is_available": staticmethod(lambda: True),
                "get_device_properties": staticmethod(lambda idx: mock_props),
            })(),
        })()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = get_vram_gb()
            assert abs(result - 128.0) < 0.01

    def test_proc_meminfo_fallback(self):
        """When torch is unavailable, falls back to /proc/meminfo."""
        meminfo = "MemTotal:       131072000 kB\nMemFree:        65536000 kB\n"
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.open", mock_open(read_data=meminfo)):
                result = get_vram_gb()
                expected = 131072000 / (1024**2)  # ~125 GB
                assert abs(result - expected) < 0.01

    def test_returns_zero_on_failure(self):
        """When both detection methods fail, returns 0.0."""
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.open", side_effect=OSError("no /proc/meminfo")):
                result = get_vram_gb()
                assert result == 0.0

    def test_torch_import_error_falls_back(self):
        """When torch import raises, falls back to /proc/meminfo."""
        meminfo = "MemTotal:       33554432 kB\nMemFree:        16777216 kB\n"

        def raise_import(*args, **kwargs):
            raise ImportError("No module named 'torch'")

        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.open", mock_open(read_data=meminfo)):
                result = get_vram_gb()
                expected = 33554432 / (1024**2)  # ~32 GB
                assert abs(result - expected) < 0.01

    def test_torch_cuda_not_available(self):
        """When torch.cuda.is_available() returns False, falls back."""
        meminfo = "MemTotal:       8388608 kB\nMemFree:        4194304 kB\n"
        mock_torch = type("MockTorch", (), {
            "cuda": type("Cuda", (), {
                "is_available": staticmethod(lambda: False),
            })(),
        })()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("builtins.open", mock_open(read_data=meminfo)):
                result = get_vram_gb()
                expected = 8388608 / (1024**2)  # ~8 GB
                assert abs(result - expected) < 0.01
