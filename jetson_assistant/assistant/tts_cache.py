"""LRU cache for TTS audio — avoids re-synthesizing repeated phrases."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CachedAudio:
    audio: np.ndarray
    sample_rate: int


class TTSCache:
    """Thread-safe LRU cache for synthesized audio clips."""

    def __init__(self, max_entries: int = 64, max_text_len: int = 80):
        self._max_entries = max_entries
        self._max_text_len = max_text_len
        self._cache: OrderedDict[str, CachedAudio] = OrderedDict()

    @staticmethod
    def _key(text: str, voice: str, lang: str) -> str:
        raw = f"{text}|{voice}|{lang}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, text: str, voice: str, lang: str) -> Optional[CachedAudio]:
        if len(text) > self._max_text_len:
            return None
        key = self._key(text, voice, lang)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, voice: str, lang: str,
            audio: np.ndarray, sample_rate: int) -> None:
        if len(text) > self._max_text_len:
            return
        key = self._key(text, voice, lang)
        self._cache[key] = CachedAudio(audio=audio, sample_rate=sample_rate)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
