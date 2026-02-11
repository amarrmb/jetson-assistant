"""
MultiWatchMonitor — concurrent camera monitoring with VLM-based condition checking.

Supports multiple simultaneous watches on different cameras. Each watch runs
in its own thread with confidence voting. Continuous monitoring with cooldown
(not one-shot like the original VisionMonitor).
"""

import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from jetson_speech.assistant.vision import WatchCondition


@dataclass
class WatchTask:
    """A single camera watch with its thread and control."""

    camera_name: str
    condition: WatchCondition
    thread: threading.Thread
    stop_event: threading.Event
    cooldown_s: float = 60.0


class MultiWatchMonitor:
    """
    Concurrent camera monitoring with VLM-based condition checking.

    Each camera can have at most one active watch condition. Starting a new
    watch on a camera replaces the previous one. Watches are continuous:
    after detection, enter cooldown then resume (not one-shot).

    Usage:
        monitor = MultiWatchMonitor(
            camera_pool=pool,
            check_fn=llm.check_condition,
            on_detected=callback,
            can_speak_fn=lambda: state == IDLE,
        )
        monitor.start_watching("garage", condition)
        monitor.start_watching("front_door", condition2)
        monitor.list_watches()
        monitor.stop_watching("garage")
        monitor.stop_all()
    """

    def __init__(
        self,
        camera_pool,
        check_fn: Callable[[str, str], bool],
        on_detected: Callable[[str, WatchCondition, Optional[str]], None],
        can_speak_fn: Callable[[], bool],
        poll_interval: float = 10.0,
        confidence_threshold: int = 2,
        vote_window: int = 3,
        cooldown_s: float = 60.0,
    ):
        """
        Args:
            camera_pool: CameraPool instance for frame capture.
            check_fn: VLM binary check function (prompt, image_b64) -> bool.
            on_detected: Callback(camera_name, condition, frame_b64) when detected.
            can_speak_fn: Returns True when assistant is idle and can speak.
            poll_interval: Seconds between VLM checks per camera.
            confidence_threshold: Min positive votes to trigger detection.
            vote_window: Sliding window size for confidence voting.
            cooldown_s: Seconds to wait after detection before resuming.
        """
        self._camera_pool = camera_pool
        self._check_fn = check_fn
        self._on_detected = on_detected
        self._can_speak_fn = can_speak_fn
        self._poll_interval = poll_interval
        self._confidence_threshold = confidence_threshold
        self._vote_window = vote_window
        self._cooldown_s = cooldown_s

        self._watches: dict[str, WatchTask] = {}
        self._watches_lock = threading.Lock()

    def start_watching(self, camera_name: str, condition: WatchCondition) -> str:
        """Start watching a camera for a condition.

        Replaces any existing watch on the same camera.

        Returns:
            Confirmation message.
        """
        if not self._camera_pool.has(camera_name):
            return f"Camera '{camera_name}' not found."

        # Stop existing watch on this camera
        with self._watches_lock:
            if camera_name in self._watches:
                self._stop_task(camera_name)

        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._monitor_loop,
            args=(camera_name, condition, stop_event),
            daemon=True,
            name=f"watch-{camera_name}",
        )

        task = WatchTask(
            camera_name=camera_name,
            condition=condition,
            thread=thread,
            stop_event=stop_event,
            cooldown_s=self._cooldown_s,
        )

        with self._watches_lock:
            self._watches[camera_name] = task

        thread.start()
        print(
            f"MultiWatch: watching '{camera_name}' for '{condition.description}'",
            file=sys.stderr,
        )
        return f"Watching {camera_name} for: {condition.description}."

    def stop_watching(self, camera_name: Optional[str] = None) -> str:
        """Stop watching a specific camera or all cameras.

        Args:
            camera_name: Camera to stop, or None to stop all.

        Returns:
            Confirmation message.
        """
        if camera_name is None or camera_name == "all":
            return self.stop_all()

        with self._watches_lock:
            if camera_name not in self._watches:
                return f"Not watching '{camera_name}'."
            self._stop_task(camera_name)

        return f"Stopped watching {camera_name}."

    def stop_all(self) -> str:
        """Stop all active watches."""
        with self._watches_lock:
            names = list(self._watches.keys())
            for name in names:
                self._stop_task(name)

        if names:
            return f"Stopped watching {', '.join(names)}."
        return "No active watches to stop."

    def list_watches(self) -> list[dict]:
        """Return info about all active watches."""
        with self._watches_lock:
            result = []
            for name, task in self._watches.items():
                result.append({
                    "camera": name,
                    "condition": task.condition.description,
                    "active": task.thread.is_alive(),
                })
            return result

    @property
    def active_watches(self) -> dict[str, WatchTask]:
        """Return dict of currently active watch tasks."""
        with self._watches_lock:
            return {k: v for k, v in self._watches.items() if v.thread.is_alive()}

    def _stop_task(self, camera_name: str) -> None:
        """Stop and remove a watch task. Must be called with _watches_lock held."""
        task = self._watches.pop(camera_name, None)
        if task is None:
            return
        task.stop_event.set()
        task.thread.join(timeout=5.0)
        print(f"MultiWatch: stopped '{camera_name}'", file=sys.stderr)

    def _monitor_loop(
        self,
        camera_name: str,
        condition: WatchCondition,
        stop_event: threading.Event,
    ) -> None:
        """Main monitor loop for a single camera watch.

        Continuously polls VLM. After detection, enters cooldown then resumes.
        """
        votes: list[bool] = []

        while not stop_event.is_set():
            # Capture frame
            frame_b64 = self._camera_pool.capture_base64(camera_name)

            if frame_b64 is None:
                # Camera not available, wait and retry
                stop_event.wait(self._poll_interval)
                continue

            # VLM inference
            try:
                detected = self._check_fn(condition.prompt, frame_b64)
            except Exception as e:
                print(f"MultiWatch [{camera_name}]: check error: {e}", file=sys.stderr)
                detected = False

            # Sliding window confidence voting
            votes.append(detected)
            if len(votes) > self._vote_window:
                votes = votes[-self._vote_window:]

            positive = sum(votes)
            print(
                f"MultiWatch [{camera_name}]: confidence {positive}/{len(votes)} "
                f"(need {self._confidence_threshold}/{self._vote_window})",
                file=sys.stderr,
            )

            if positive >= self._confidence_threshold and self._can_speak_fn():
                # Condition detected — fire callback
                self._on_detected(camera_name, condition, frame_b64)

                # Reset votes and enter cooldown
                votes.clear()
                print(
                    f"MultiWatch [{camera_name}]: cooldown {self._cooldown_s}s",
                    file=sys.stderr,
                )
                stop_event.wait(self._cooldown_s)
                if stop_event.is_set():
                    return
                print(f"MultiWatch [{camera_name}]: resuming watch", file=sys.stderr)

            # Sleep between polls (interruptible)
            stop_event.wait(self._poll_interval)
