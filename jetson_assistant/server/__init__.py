"""
FastAPI server for Jetson Assistant.
"""

from jetson_assistant.server.app import create_app

__all__ = ["create_app"]
