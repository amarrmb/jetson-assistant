"""
Pipelined streaming for real-time TTS playback.

Implements a producer-consumer pattern where audio is generated
ahead of playback for smoother streaming.
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from jetson_speech.tts.base import SynthesisResult


@dataclass
class StreamChunk:
    """A chunk of streamed audio."""

    index: int
    audio: np.ndarray
    sample_rate: int
    text: str
    is_last: bool = False
    error: Exception | None = None


class StreamingPipeline:
    """
    Pipelined streaming for TTS.

    Uses a producer-consumer pattern to generate audio ahead
    of playback, reducing perceived latency.
    """

    def __init__(
        self,
        synthesize_fn: Callable[[str], SynthesisResult],
        buffer_size: int = 3,
    ):
        """
        Initialize the streaming pipeline.

        Args:
            synthesize_fn: Function to synthesize a single chunk
            buffer_size: Number of chunks to buffer ahead
        """
        self.synthesize_fn = synthesize_fn
        self.buffer_size = buffer_size
        self._audio_queue: queue.Queue[StreamChunk | None] = queue.Queue(maxsize=buffer_size)
        self._producer_thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stop_flag = threading.Event()

    def stream(self, chunks: list[str]) -> Iterator[StreamChunk]:
        """
        Stream audio chunks with pipelined generation.

        Args:
            chunks: List of text chunks to synthesize

        Yields:
            StreamChunk objects with audio data
        """
        if not chunks:
            return

        self._error = None
        self._stop_flag.clear()

        # Start producer thread
        self._producer_thread = threading.Thread(
            target=self._producer,
            args=(chunks,),
            daemon=True,
        )
        self._producer_thread.start()

        # Yield chunks as they become available
        try:
            while True:
                chunk = self._audio_queue.get()

                if chunk is None:
                    # End of stream
                    break

                if chunk.error:
                    raise chunk.error

                yield chunk

        finally:
            # Signal producer to stop if still running
            self._stop_flag.set()

            # Wait for producer to finish
            if self._producer_thread and self._producer_thread.is_alive():
                self._producer_thread.join(timeout=5.0)

    def _producer(self, chunks: list[str]) -> None:
        """Producer thread: generate audio chunks."""
        try:
            for i, text in enumerate(chunks):
                if self._stop_flag.is_set():
                    break

                # Synthesize chunk
                result = self.synthesize_fn(text)

                # Put in queue (blocks if full)
                self._audio_queue.put(
                    StreamChunk(
                        index=i,
                        audio=result.audio,
                        sample_rate=result.sample_rate,
                        text=text,
                        is_last=(i == len(chunks) - 1),
                    )
                )

        except Exception as e:
            # Send error to consumer
            self._audio_queue.put(
                StreamChunk(
                    index=-1,
                    audio=np.array([], dtype=np.int16),
                    sample_rate=0,
                    text="",
                    error=e,
                )
            )

        finally:
            # Signal end of stream
            self._audio_queue.put(None)


class StreamingPlayer:
    """
    Real-time audio player for streaming TTS.

    Plays audio chunks as they arrive while buffering
    for smooth playback.
    """

    def __init__(self, buffer_ahead: int = 2):
        """
        Initialize the streaming player.

        Args:
            buffer_ahead: Number of chunks to buffer before starting playback
        """
        self.buffer_ahead = buffer_ahead
        self._play_queue: queue.Queue[StreamChunk | None] = queue.Queue()
        self._player_thread: threading.Thread | None = None
        self._all_chunks: list[StreamChunk] = []

    def play_stream(
        self,
        stream: Iterator[StreamChunk],
        on_chunk_played: Callable[[StreamChunk], None] | None = None,
    ) -> list[StreamChunk]:
        """
        Play a stream of audio chunks.

        Args:
            stream: Iterator of StreamChunk objects
            on_chunk_played: Callback when a chunk finishes playing

        Returns:
            List of all played chunks
        """
        from jetson_speech.core.audio import play_audio

        self._all_chunks = []
        buffered = 0

        # Start player thread
        self._player_thread = threading.Thread(
            target=self._player,
            args=(play_audio, on_chunk_played),
            daemon=True,
        )
        self._player_thread.start()

        try:
            for chunk in stream:
                self._all_chunks.append(chunk)
                self._play_queue.put(chunk)
                buffered += 1

                # Start playing after initial buffer
                if buffered == self.buffer_ahead:
                    pass  # Player thread is already running

        finally:
            # Signal end of stream
            self._play_queue.put(None)

            # Wait for player to finish
            if self._player_thread:
                self._player_thread.join()

        return self._all_chunks

    def _player(
        self,
        play_fn: Callable[[np.ndarray, int], None],
        on_chunk_played: Callable[[StreamChunk], None] | None,
    ) -> None:
        """Player thread: play audio chunks as they arrive."""
        while True:
            chunk = self._play_queue.get()

            if chunk is None:
                break

            # Play the chunk
            play_fn(chunk.audio, chunk.sample_rate)

            # Callback
            if on_chunk_played:
                on_chunk_played(chunk)


def stream_and_play(
    chunks: list[str],
    synthesize_fn: Callable[[str], SynthesisResult],
    buffer_size: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[SynthesisResult]:
    """
    High-level function to stream and play TTS output.

    Args:
        chunks: List of text chunks to synthesize
        synthesize_fn: Function to synthesize a single chunk
        buffer_size: Number of chunks to buffer ahead
        on_progress: Callback with (current, total) progress

    Returns:
        List of SynthesisResult objects
    """
    pipeline = StreamingPipeline(synthesize_fn, buffer_size=buffer_size)
    player = StreamingPlayer(buffer_ahead=min(2, buffer_size))

    def progress_callback(chunk: StreamChunk) -> None:
        if on_progress:
            on_progress(chunk.index + 1, len(chunks))

    played_chunks = player.play_stream(
        pipeline.stream(chunks),
        on_chunk_played=progress_callback,
    )

    # Convert to SynthesisResult
    results = []
    for chunk in played_chunks:
        results.append(
            SynthesisResult(
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                text=chunk.text,
            )
        )

    return results
