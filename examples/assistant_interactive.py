#!/usr/bin/env python3
"""
Interactive Voice Assistant Example

This example shows programmatic control of the assistant:
- Make the assistant speak
- Listen for user input
- Ask questions and get responses
- Run in background

Requirements:
    pip install jetson-assistant[assistant,qwen,whisper]
    ollama pull llama3.2:3b

Usage:
    python assistant_interactive.py
"""

import time

from jetson_assistant import Engine
from jetson_assistant.assistant import VoiceAssistant, AssistantConfig


def main():
    print("Initializing Interactive Voice Assistant...")

    # Create engine
    engine = Engine()
    engine.load_tts_backend("qwen", model_size="0.6B")
    engine.load_stt_backend("whisper", model_size="base")

    # Configure without wake word for interactive use
    config = AssistantConfig(
        wake_word_backend="energy",  # No wake word needed
        llm_backend="ollama",
        llm_model="llama3.2:3b",
        tts_voice="Chelsie",
        play_chimes=True,
        verbose=True,
    )

    assistant = VoiceAssistant(engine, config)

    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("=" * 50)

    # Demo 1: Make the assistant speak
    print("\n1. Making assistant speak...")
    assistant.say("Hello! I'm your interactive voice assistant. Let me show you what I can do.")
    time.sleep(0.5)

    # Demo 2: Listen for user input
    print("\n2. Listening for your input (speak now)...")
    text = assistant.listen_once(timeout=8.0)
    if text:
        print(f"   You said: {text}")
    else:
        print("   (No speech detected)")

    # Demo 3: Ask a question
    print("\n3. Asking a question...")
    assistant.say("What's your favorite color?")
    answer = assistant.listen_once(timeout=8.0)
    if answer:
        print(f"   You answered: {answer}")
        assistant.say(f"Nice! {answer} is a great choice!")
    else:
        assistant.say("I didn't catch that, but that's okay!")

    # Demo 4: Run in background
    print("\n4. Running in background mode...")
    print("   Say something to interact (10 seconds)...")

    # Start in background
    thread = assistant.run_async()

    # Do other things while assistant runs
    for i in range(10):
        print(f"   Main thread working... {10-i}s remaining")
        time.sleep(1)

    # Stop background assistant
    assistant.stop()
    thread.join()

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)

    # Show conversation history
    history = assistant.get_conversation_history()
    if history:
        print("\nConversation history:")
        for msg in history:
            role = "You" if msg["role"] == "user" else "Assistant"
            print(f"  {role}: {msg['content']}")


if __name__ == "__main__":
    main()
