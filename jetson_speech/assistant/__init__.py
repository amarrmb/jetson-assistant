"""
Voice Assistant module for jetson-speech.

Build your own Alexa-like voice assistant using local TTS, STT, and LLM.
"""

from jetson_speech.assistant.core import VoiceAssistant, AssistantConfig

__all__ = ["VoiceAssistant", "AssistantConfig"]
