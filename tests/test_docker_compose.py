"""Tests for docker-compose.yml validity and structure."""

from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose():
    """Load docker-compose.yml once."""
    with open(COMPOSE_FILE) as f:
        return yaml.safe_load(f)


class TestDockerCompose:
    def test_yaml_is_valid(self, compose):
        """docker-compose.yml parses as valid YAML."""
        assert compose is not None
        assert "services" in compose

    def test_required_services_exist(self, compose):
        """All expected services are defined."""
        services = compose["services"]
        for name in ("vllm", "vllm-thor-bf16", "vllm-orin", "whisper"):
            assert name in services, f"Missing service: {name}"

    def test_all_services_have_healthchecks(self, compose):
        """Every service must define a healthcheck."""
        for name, svc in compose["services"].items():
            assert "healthcheck" in svc, f"Service '{name}' is missing a healthcheck"
            assert "test" in svc["healthcheck"], f"Service '{name}' healthcheck has no test"

    def test_vlm_services_use_port_8001(self, compose):
        """All VLM services must serve on port 8001."""
        for name in ("vllm", "vllm-thor-bf16", "vllm-orin"):
            cmd = compose["services"][name]["command"]
            assert "--port 8001" in cmd, f"Service '{name}' not using port 8001"

    def test_whisper_uses_port_8002(self, compose):
        """Whisper STT must serve on port 8002."""
        cmd = compose["services"]["whisper"]["command"]
        assert "--port 8002" in cmd

    def test_profiles(self, compose):
        """vllm has no profile (default), others have expected profiles."""
        services = compose["services"]
        assert "profiles" not in services["vllm"], "Default vllm should have no profile"
        assert "thor-bf16" in services["vllm-thor-bf16"]["profiles"]
        assert "orin" in services["vllm-orin"]["profiles"]
        assert "gpu-stt" in services["whisper"]["profiles"]

    def test_all_services_use_nvidia_runtime(self, compose):
        """All services should use the nvidia runtime (via x-common anchor)."""
        for name, svc in compose["services"].items():
            assert svc.get("runtime") == "nvidia", (
                f"Service '{name}' missing runtime: nvidia"
            )

    def test_all_services_use_host_network(self, compose):
        """All services should use host networking."""
        for name, svc in compose["services"].items():
            assert svc.get("network_mode") == "host", (
                f"Service '{name}' missing network_mode: host"
            )
