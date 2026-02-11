#!/usr/bin/env python3
"""
Vision Voice Assistant Example

This example shows how to create a voice assistant with camera vision.
The assistant can "see" through a USB camera and answer questions about
what it observes, using a vision-language model (VLM) via Ollama.

Requirements:
    pip install jetson-assistant[assistant,whisper,vision]
    ollama pull moondream  # or llama3.2-vision:11b

Usage:
    python assistant_vision.py
    python assistant_vision.py --model llama3.2-vision:11b --no-wake
    python assistant_vision.py --camera 1 --server
"""

import argparse

from jetson_assistant.assistant import VoiceAssistant, AssistantConfig


def main():
    parser = argparse.ArgumentParser(description="Vision Voice Assistant")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--model", type=str, default="moondream", help="VLM model name")
    parser.add_argument("--no-wake", action="store_true", help="Skip wake word (always listening)")
    parser.add_argument("--server", action="store_true", help="Use speech server for TTS/STT")
    parser.add_argument("--server-port", type=int, default=8080, help="Speech server port")
    args = parser.parse_args()

    print("Vision Voice Assistant")
    print("=" * 40)
    print(f"Camera: device {args.camera}")
    print(f"Model: {args.model}")
    print()

    engine = None
    if not args.server:
        from jetson_assistant.core.engine import Engine

        engine = Engine()
        print("Loading TTS (piper)...")
        engine.load_tts_backend("piper")
        print("Loading STT (whisper)...")
        engine.load_stt_backend("whisper", model_size="base", device="cpu")

    config = AssistantConfig(
        use_server=args.server,
        server_port=args.server_port,
        wake_word="hey_jarvis" if not args.no_wake else "energy",
        wake_word_backend="openwakeword" if not args.no_wake else "energy",
        llm_backend="ollama",
        llm_model=args.model,
        tts_voice="en_US-amy-medium",
        verbose=True,
        vision_enabled=True,
        camera_device=args.camera,
    )

    assistant = VoiceAssistant(engine, config)

    print("\nReady! Try asking: 'What do you see?'")
    print("Press Ctrl+C to stop.\n")

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
