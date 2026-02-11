#!/usr/bin/env python3
"""
Lightweight Voice Assistant for Resource-Constrained Devices

This example is optimized for devices with limited RAM like Jetson Orin Nano 4GB.
Uses:
- Piper TTS (CPU, fast, small memory)
- Whisper tiny (smallest model)
- Phi3 mini (small but capable LLM)

Requirements:
    pip install jetson-speech[assistant,piper,whisper]
    ollama pull phi3:mini

Usage:
    python assistant_lightweight.py
"""

from jetson_speech import Engine
from jetson_speech.assistant import VoiceAssistant, AssistantConfig


def main():
    print("=" * 50)
    print("Lightweight Voice Assistant")
    print("(Optimized for Jetson Orin Nano 4GB)")
    print("=" * 50)

    # Create engine
    engine = Engine()

    # Use lightweight backends
    print("\nLoading Piper TTS (CPU, lightweight)...")
    engine.load_tts_backend("piper", voice="en_US-lessac-medium")

    print("Loading Whisper tiny (smallest model)...")
    engine.load_stt_backend("whisper", model_size="tiny")

    # Configure for minimal resource usage
    config = AssistantConfig(
        # Wake word
        wake_word="hey_jarvis",

        # Use smaller LLM
        llm_backend="ollama",
        llm_model="phi3:mini",  # Smaller than llama3.2

        # Piper voice
        tts_backend="piper",
        tts_voice="en_US-lessac-medium",

        # Shorter timeout for faster response
        silence_timeout_ms=1000,
        max_listen_time_s=8.0,

        # Disable chimes to save memory
        play_chimes=False,

        verbose=True,
    )

    assistant = VoiceAssistant(engine, config)

    print("\nAssistant ready! Say 'hey jarvis' to activate.")
    print("Press Ctrl+C to stop.\n")

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
