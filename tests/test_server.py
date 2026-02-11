"""
Tests for the FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from jetson_speech.server.app import create_app

    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "is_jetson" in data

    def test_info_endpoint(self, client):
        """Test info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert "tts" in data
        assert "stt" in data


class TestTTSEndpoints:
    """Test TTS API endpoints."""

    def test_list_backends(self, client):
        """Test listing TTS backends."""
        response = client.get("/tts/backends")
        assert response.status_code == 200

        data = response.json()
        assert "backends" in data
        assert isinstance(data["backends"], list)

    def test_synthesize_without_backend(self, client):
        """Test synthesis fails without backend loaded."""
        response = client.post(
            "/tts/synthesize",
            json={"text": "Hello world"},
        )
        assert response.status_code == 400

    def test_get_voices_without_backend(self, client):
        """Test getting voices fails without backend."""
        response = client.get("/tts/voices")
        assert response.status_code == 400


class TestSTTEndpoints:
    """Test STT API endpoints."""

    def test_list_backends(self, client):
        """Test listing STT backends."""
        response = client.get("/stt/backends")
        assert response.status_code == 200

        data = response.json()
        assert "backends" in data
        assert isinstance(data["backends"], list)

    def test_get_languages_without_backend(self, client):
        """Test getting languages fails without backend."""
        response = client.get("/stt/languages")
        assert response.status_code == 400
