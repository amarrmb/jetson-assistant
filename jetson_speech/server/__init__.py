"""
FastAPI server for Jetson Speech.
"""

from jetson_speech.server.app import create_app

__all__ = ["create_app"]
