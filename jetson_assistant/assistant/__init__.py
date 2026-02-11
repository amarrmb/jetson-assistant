"""
Voice Assistant module for jetson-assistant.

Build your own Alexa-like voice assistant using local TTS, STT, and LLM.
"""

from jetson_assistant.assistant.core import VoiceAssistant, AssistantConfig

__all__ = ["VoiceAssistant", "AssistantConfig"]
