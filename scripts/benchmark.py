#!/usr/bin/env python3
"""
Benchmark script for jetson-assistant components.
Tests TTS, STT, LLM individually and measures performance.
"""

import sys
import time

def test_tts():
    """Test TTS standalone performance."""
    print("=" * 60)
    print("TEST 1: TTS STANDALONE PERFORMANCE")
    print("=" * 60)

    from jetson_assistant.tts.qwen import QwenBackend

    tts = QwenBackend()
    print("Loading TTS model...")
    load_start = time.perf_counter()
    tts.load()
    load_time = time.perf_counter() - load_start
    print(f"Model load time: {load_time:.2f}s")
    print()

    # Test sentences of varying lengths
    test_sentences = [
        ("Short (5 words)", "Hello, how are you today?"),
        ("Medium (10 words)", "The quick brown fox jumps over the lazy dog today."),
        ("Long (18 words)", "Invoker is considered one of the most powerful heroes in Dota 2 due to his versatility."),
    ]

    print("Warming up model...")
    _ = tts.synthesize("Test warmup sentence.", voice="serena")
    print()

    results = []
    for name, sentence in test_sentences:
        word_count = len(sentence.split())
        char_count = len(sentence)

        start = time.perf_counter()
        result = tts.synthesize(sentence, voice="serena")
        elapsed = time.perf_counter() - start

        audio_duration = len(result.audio) / result.sample_rate
        rtf = elapsed / audio_duration

        print(f"{name}:")
        print(f"  Text: {sentence}")
        print(f"  Words: {word_count}, Chars: {char_count}")
        print(f"  TTS time: {elapsed:.2f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  RTF: {rtf:.2f}x (lower is better, <1 = faster than realtime)")
        print()

        results.append({
            "name": name,
            "words": word_count,
            "chars": char_count,
            "tts_time": elapsed,
            "audio_duration": audio_duration,
            "rtf": rtf,
        })

    return results


def test_llm():
    """Test LLM standalone performance."""
    print("=" * 60)
    print("TEST 2: LLM STANDALONE PERFORMANCE")
    print("=" * 60)

    from jetson_assistant.assistant.llm import OllamaLLM

    llm = OllamaLLM(model="phi3:mini")
    print(f"Using model: phi3:mini")
    print(f"System prompt: {llm.system_prompt}")
    print()

    test_prompts = [
        "What is 2 plus 2?",
        "Who is the strongest hero in Dota 2?",
        "What time is it?",
    ]

    results = []
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")

        # Test streaming
        start = time.perf_counter()
        sentences = list(llm.generate_stream(prompt))
        elapsed = time.perf_counter() - start

        full_response = " ".join(sentences)
        word_count = len(full_response.split())

        print(f"  Response: {full_response}")
        print(f"  Words: {word_count}")
        print(f"  LLM time: {elapsed:.2f}s")
        print(f"  First sentence: {sentences[0] if sentences else 'N/A'}")
        print()

        results.append({
            "prompt": prompt,
            "response": full_response,
            "words": word_count,
            "time": elapsed,
        })

    return results


def test_comparison_with_standalone():
    """Compare our TTS with standalone qwen-tts.sh."""
    print("=" * 60)
    print("TEST 3: COMPARISON - jetson-assistant vs standalone")
    print("=" * 60)

    import subprocess
    import os

    test_text = "Axe is known as The Reaper and often emerges victorious in battle."

    # Test our implementation
    print("Testing jetson-assistant TTS...")
    from jetson_assistant.tts.qwen import QwenBackend

    tts = QwenBackend()
    tts.load()

    # Warmup
    _ = tts.synthesize("Warmup.", voice="vivian")

    start = time.perf_counter()
    result = tts.synthesize(test_text, voice="vivian")
    our_time = time.perf_counter() - start
    our_audio_duration = len(result.audio) / result.sample_rate

    print(f"  jetson-assistant TTS time: {our_time:.2f}s")
    print(f"  Audio duration: {our_audio_duration:.2f}s")
    print(f"  RTF: {our_time/our_audio_duration:.2f}x")
    print()

    # Note about standalone comparison
    print("Note: To compare with standalone qwen-tts.sh, run:")
    print(f'  cd ~/Qwen3-TTS && ./qwen-tts.sh -s vivian "{test_text}"')
    print()

    return {
        "text": test_text,
        "our_time": our_time,
        "our_audio_duration": our_audio_duration,
        "our_rtf": our_time / our_audio_duration,
    }


def main():
    print("\n" + "=" * 60)
    print("JETSON-ASSISTANT BENCHMARK")
    print("=" * 60 + "\n")

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        tts_results = test_tts()
    except Exception as e:
        print(f"TTS test failed: {e}")
        tts_results = None

    print()

    try:
        llm_results = test_llm()
    except Exception as e:
        print(f"LLM test failed: {e}")
        llm_results = None

    print()

    try:
        comparison = test_comparison_with_standalone()
    except Exception as e:
        print(f"Comparison test failed: {e}")
        comparison = None

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if tts_results:
        print("\nTTS Performance:")
        for r in tts_results:
            print(f"  {r['name']}: {r['tts_time']:.2f}s for {r['words']} words (RTF: {r['rtf']:.2f}x)")

        avg_rtf = sum(r['rtf'] for r in tts_results) / len(tts_results)
        print(f"  Average RTF: {avg_rtf:.2f}x")

    if llm_results:
        print("\nLLM Performance:")
        for r in llm_results:
            print(f"  '{r['prompt'][:30]}...' -> {r['words']} words in {r['time']:.2f}s")

    print("\nRecommendations:")
    if tts_results:
        avg_rtf = sum(r['rtf'] for r in tts_results) / len(tts_results)
        if avg_rtf > 2.0:
            print("  - TTS is slow (RTF > 2x). Consider shorter responses.")
        elif avg_rtf > 1.0:
            print("  - TTS is acceptable (1-2x RTF). Room for optimization.")
        else:
            print("  - TTS is fast (<1x RTF). Generating faster than realtime!")

    if llm_results:
        avg_words = sum(r['words'] for r in llm_results) / len(llm_results)
        if avg_words > 15:
            print(f"  - LLM responses too long (avg {avg_words:.0f} words). Reduce token limit.")
        elif avg_words < 5:
            print(f"  - LLM responses very short (avg {avg_words:.0f} words). May be cut off.")
        else:
            print(f"  - LLM response length OK (avg {avg_words:.0f} words).")


if __name__ == "__main__":
    main()
