"""Tests for TTS audio cache with LRU eviction."""

import numpy as np

from jetson_assistant.assistant.tts_cache import TTSCache


def test_cache_miss_returns_none():
    cache = TTSCache(max_entries=10)
    assert cache.get("hello", "af_heart", "en") is None


def test_cache_hit_returns_audio():
    cache = TTSCache(max_entries=10)
    audio = np.zeros(16000, dtype=np.int16)
    cache.put("hello", "af_heart", "en", audio, 24000)
    result = cache.get("hello", "af_heart", "en")
    assert result is not None
    assert result.sample_rate == 24000
    assert np.array_equal(result.audio, audio)


def test_cache_evicts_oldest():
    cache = TTSCache(max_entries=2)
    a1 = np.zeros(100, dtype=np.int16)
    a2 = np.zeros(200, dtype=np.int16)
    a3 = np.zeros(300, dtype=np.int16)
    cache.put("one", "v", "en", a1, 24000)
    cache.put("two", "v", "en", a2, 24000)
    cache.put("three", "v", "en", a3, 24000)  # evicts "one"
    assert cache.get("one", "v", "en") is None
    assert cache.get("two", "v", "en") is not None


def test_cache_skips_long_text():
    cache = TTSCache(max_entries=10, max_text_len=50)
    audio = np.zeros(16000, dtype=np.int16)
    long_text = "x" * 100
    cache.put(long_text, "v", "en", audio, 24000)
    assert cache.get(long_text, "v", "en") is None  # not cached


def test_cache_lru_order_on_get():
    """Accessing an entry should refresh it so it is not evicted next."""
    cache = TTSCache(max_entries=2)
    a1 = np.zeros(100, dtype=np.int16)
    a2 = np.zeros(200, dtype=np.int16)
    a3 = np.zeros(300, dtype=np.int16)
    cache.put("one", "v", "en", a1, 24000)
    cache.put("two", "v", "en", a2, 24000)
    # Access "one" to make it most-recently-used
    cache.get("one", "v", "en")
    # Insert "three" — should evict "two" (oldest), not "one"
    cache.put("three", "v", "en", a3, 24000)
    assert cache.get("one", "v", "en") is not None
    assert cache.get("two", "v", "en") is None


def test_cache_different_voices_are_separate():
    """Same text with different voice/lang should be separate cache entries."""
    cache = TTSCache(max_entries=10)
    audio_a = np.ones(100, dtype=np.int16)
    audio_b = np.ones(200, dtype=np.int16) * 2
    cache.put("hello", "af_heart", "en", audio_a, 24000)
    cache.put("hello", "am_adam", "en", audio_b, 22050)
    result_a = cache.get("hello", "af_heart", "en")
    result_b = cache.get("hello", "am_adam", "en")
    assert result_a is not None
    assert result_b is not None
    assert result_a.sample_rate == 24000
    assert result_b.sample_rate == 22050
    assert len(result_a.audio) == 100
    assert len(result_b.audio) == 200


def test_cache_overwrite_same_key():
    """Putting the same key again should overwrite the cached value."""
    cache = TTSCache(max_entries=10)
    audio1 = np.zeros(100, dtype=np.int16)
    audio2 = np.ones(200, dtype=np.int16)
    cache.put("hello", "v", "en", audio1, 24000)
    cache.put("hello", "v", "en", audio2, 16000)
    result = cache.get("hello", "v", "en")
    assert result is not None
    assert result.sample_rate == 16000
    assert np.array_equal(result.audio, audio2)
