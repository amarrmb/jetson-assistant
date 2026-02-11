#!/usr/bin/env python3
"""
Basic Voice Assistant Example

This example shows how to create a simple voice assistant
with wake word detection, STT, LLM, and TTS.

Requirements:
    pip install jetson-assistant[assistant,qwen,whisper]
    ollama pull llama3.2:3b

Usage:
    python assistant_basic.py
"""

from jetson_assistant import Engine
from jetson_assistant.assistant import VoiceAssistant, AssistantConfig


def main():
    # Create engine and load backends
    print("Initializing Jetson Assistant Engine...")
    engine = Engine()

    print("Loading TTS (Qwen)...")
    engine.load_tts_backend("qwen", model_size="0.6B")

    print("Loading STT (Whisper)...")
    engine.load_stt_backend("whisper", model_size="base")

    # Configure the assistant
    config = AssistantConfig(
        # Wake word
        wake_word="hey_jarvis",
        wake_word_backend="openwakeword",

        # LLM (local with Ollama)
        llm_backend="ollama",
        llm_model="llama3.2:3b",

        # TTS settings
        tts_voice="Chelsie",
        tts_language="English",

        # Show timing info
        verbose=True,

        # Audio feedback
        play_chimes=True,
    )

    # Create and run the assistant
    print("\n" + "=" * 50)
    print("Voice Assistant Ready!")
    print("=" * 50)
    print(f"\nSay '{config.wake_word.replace('_', ' ')}' to activate.")
    print("Press Ctrl+C to stop.\n")

    assistant = VoiceAssistant(engine, config)

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
