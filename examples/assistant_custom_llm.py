#!/usr/bin/env python3
"""
Custom LLM Voice Assistant Example

This example shows how to use different LLM backends:
- Ollama (local)
- OpenAI (cloud)
- Anthropic Claude (cloud)
- Simple (rule-based, no LLM needed)

Requirements:
    pip install jetson-assistant[assistant,qwen,whisper]

    # For Ollama:
    ollama pull llama3.2:3b

    # For OpenAI:
    export OPENAI_API_KEY=sk-...
    pip install openai

    # For Anthropic:
    export ANTHROPIC_API_KEY=sk-...
    pip install anthropic

Usage:
    python assistant_custom_llm.py --llm ollama
    python assistant_custom_llm.py --llm openai
    python assistant_custom_llm.py --llm simple
"""

import argparse

from jetson_assistant import Engine
from jetson_assistant.assistant import VoiceAssistant, AssistantConfig
from jetson_assistant.assistant.llm import create_llm


# Custom personalities
PERSONALITIES = {
    "default": """You are a helpful voice assistant running on an NVIDIA Jetson device.
Keep your responses concise and conversational (1-3 sentences max).
Do not use markdown formatting - speak naturally.""",

    "jarvis": """You are JARVIS, an advanced AI assistant inspired by Tony Stark's creation.
You are witty, intelligent, and occasionally sarcastic.
Address the user as "sir" or "ma'am" when appropriate.
Keep responses brief but sophisticated.""",

    "friendly": """You are a warm, friendly assistant who loves to help.
Be encouraging and positive in your responses.
Use casual, conversational language.
Keep responses short and sweet.""",

    "professional": """You are a professional executive assistant.
Be formal, precise, and efficient in your responses.
Focus on being helpful and informative.
Keep responses concise and businesslike.""",
}


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant with Custom LLM")
    parser.add_argument("--llm", choices=["ollama", "openai", "anthropic", "simple"],
                        default="ollama", help="LLM backend to use")
    parser.add_argument("--model", type=str, help="Model name (backend-specific)")
    parser.add_argument("--personality", choices=list(PERSONALITIES.keys()),
                        default="default", help="Assistant personality")
    parser.add_argument("--no-wake", action="store_true", help="Disable wake word")
    args = parser.parse_args()

    # Get model name based on backend
    if args.model:
        model = args.model
    else:
        model = {
            "ollama": "llama3.2:3b",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "simple": None,
        }[args.llm]

    print(f"Using LLM: {args.llm}" + (f" ({model})" if model else ""))
    print(f"Personality: {args.personality}")

    # Create engine
    engine = Engine()
    print("Loading TTS...")
    engine.load_tts_backend("qwen", model_size="0.6B")
    print("Loading STT...")
    engine.load_stt_backend("whisper", model_size="base")

    # Create custom LLM with personality
    llm = create_llm(
        backend=args.llm,
        model=model,
        system_prompt=PERSONALITIES[args.personality],
    )

    # Configure assistant
    config = AssistantConfig(
        wake_word="hey_jarvis" if not args.no_wake else "energy",
        wake_word_backend="openwakeword" if not args.no_wake else "energy",
        tts_voice="Chelsie",
        verbose=True,
    )

    # Create assistant with custom LLM
    assistant = VoiceAssistant(engine, config, llm=llm)

    print("\n" + "=" * 50)
    print("Voice Assistant Ready!")
    print("=" * 50)
    if args.no_wake:
        print("\nListening mode: Always on (speak to activate)")
    else:
        print("\nSay 'hey jarvis' to activate.")
    print("Press Ctrl+C to stop.\n")

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
