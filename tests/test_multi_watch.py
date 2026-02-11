"""
Tests for MultiWatchMonitor concurrent camera monitoring.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jetson_speech.assistant.vision import WatchCondition


def _make_condition(desc="test condition"):
    return WatchCondition(
        description=desc,
        prompt=f"{desc}? Answer only YES or NO.",
        announce_template=f"Alert: {desc}!",
    )


def _make_pool_with_camera(name="test_cam"):
    """Create a CameraPool with a mock camera that returns frames."""
    from jetson_speech.assistant.cameras import CameraPool

    tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump([{"name": name, "url": "usb:0"}], tmpfile)
    tmpfile.close()

    pool = CameraPool(config_path=Path(tmpfile.name))
    # Mock capture_base64 to return a fake frame
    pool.capture_base64 = MagicMock(return_value="fake_base64_frame")
    return pool, tmpfile.name


class TestMultiWatchMonitor:
    """Tests for MultiWatchMonitor."""

    def _make_monitor(self, pool=None, check_fn=None, on_detected=None,
                      poll_interval=0.1, cooldown_s=0.5):
        from jetson_speech.assistant.multi_watch import MultiWatchMonitor

        if pool is None:
            pool, self._tmpfile = _make_pool_with_camera()
        if check_fn is None:
            check_fn = MagicMock(return_value=False)
        if on_detected is None:
            on_detected = MagicMock()

        monitor = MultiWatchMonitor(
            camera_pool=pool,
            check_fn=check_fn,
            on_detected=on_detected,
            can_speak_fn=lambda: True,
            poll_interval=poll_interval,
            confidence_threshold=2,
            vote_window=3,
            cooldown_s=cooldown_s,
        )
        return monitor, check_fn, on_detected

    def _cleanup(self):
        import os
        if hasattr(self, "_tmpfile") and os.path.exists(self._tmpfile):
            os.unlink(self._tmpfile)

    def test_start_and_list(self):
        """Start a watch, verify it appears in active list."""
        monitor, _, _ = self._make_monitor()
        try:
            condition = _make_condition("door open")
            result = monitor.start_watching("test_cam", condition)

            assert "watching" in result.lower()

            watches = monitor.list_watches()
            assert len(watches) == 1
            assert watches[0]["camera"] == "test_cam"
            assert watches[0]["condition"] == "door open"

            monitor.stop_all()
        finally:
            self._cleanup()

    def test_stop_specific(self):
        """Stop one watch, verify it's removed."""
        pool, tmpfile = _make_pool_with_camera("cam1")
        pool.add("cam2", "usb:1")
        pool.capture_base64 = MagicMock(return_value="fake_frame")
        self._tmpfile = tmpfile

        monitor, _, _ = self._make_monitor(pool=pool)
        try:
            monitor.start_watching("cam1", _make_condition("cond1"))
            monitor.start_watching("cam2", _make_condition("cond2"))

            assert len(monitor.list_watches()) == 2

            result = monitor.stop_watching("cam1")
            assert "stopped" in result.lower()

            watches = monitor.list_watches()
            assert len(watches) == 1
            assert watches[0]["camera"] == "cam2"

            monitor.stop_all()
        finally:
            self._cleanup()

    def test_stop_all(self):
        """Stop all watches."""
        pool, tmpfile = _make_pool_with_camera("cam1")
        pool.add("cam2", "usb:1")
        pool.capture_base64 = MagicMock(return_value="fake_frame")
        self._tmpfile = tmpfile

        monitor, _, _ = self._make_monitor(pool=pool)
        try:
            monitor.start_watching("cam1", _make_condition("cond1"))
            monitor.start_watching("cam2", _make_condition("cond2"))

            result = monitor.stop_all()
            assert "stopped" in result.lower()
            assert len(monitor.list_watches()) == 0
        finally:
            self._cleanup()

    def test_replace_watch(self):
        """New watch on same camera replaces the old one."""
        monitor, _, _ = self._make_monitor()
        try:
            monitor.start_watching("test_cam", _make_condition("old condition"))
            monitor.start_watching("test_cam", _make_condition("new condition"))

            watches = monitor.list_watches()
            assert len(watches) == 1
            assert watches[0]["condition"] == "new condition"

            monitor.stop_all()
        finally:
            self._cleanup()

    def test_detection_callback(self):
        """Mock check_fn returning True triggers on_detected."""
        # Always return True so detection threshold is met quickly
        check_fn = MagicMock(return_value=True)
        on_detected = MagicMock()

        monitor, _, _ = self._make_monitor(
            check_fn=check_fn,
            on_detected=on_detected,
            poll_interval=0.05,
            cooldown_s=0.5,
        )
        try:
            condition = _make_condition("person detected")
            monitor.start_watching("test_cam", condition)

            # Wait for detection (2 positive votes needed at 0.05s intervals)
            time.sleep(0.5)

            assert on_detected.called
            args = on_detected.call_args
            assert args[0][0] == "test_cam"  # camera_name
            assert args[0][1].description == "person detected"  # condition

            monitor.stop_all()
        finally:
            self._cleanup()

    def test_no_false_detection(self):
        """check_fn returning False should not trigger on_detected."""
        check_fn = MagicMock(return_value=False)
        on_detected = MagicMock()

        monitor, _, _ = self._make_monitor(
            check_fn=check_fn,
            on_detected=on_detected,
            poll_interval=0.05,
        )
        try:
            monitor.start_watching("test_cam", _make_condition("nothing"))

            time.sleep(0.3)

            assert not on_detected.called

            monitor.stop_all()
        finally:
            self._cleanup()

    def test_stop_nonexistent(self):
        """Stopping a non-existent watch returns appropriate message."""
        monitor, _, _ = self._make_monitor()
        try:
            result = monitor.stop_watching("nonexistent")
            assert "not watching" in result.lower()
        finally:
            self._cleanup()

    def test_stop_all_empty(self):
        """Stopping all when no watches are active returns appropriate message."""
        monitor, _, _ = self._make_monitor()
        try:
            result = monitor.stop_all()
            assert "no active" in result.lower()
        finally:
            self._cleanup()

    def test_watch_nonexistent_camera(self):
        """Watching a camera that doesn't exist returns error."""
        monitor, _, _ = self._make_monitor()
        try:
            result = monitor.start_watching("ghost", _make_condition())
            assert "not found" in result.lower()
        finally:
            self._cleanup()

    def test_cooldown_prevents_rapid_alerts(self):
        """After detection, cooldown prevents immediate re-alert."""
        check_fn = MagicMock(return_value=True)
        on_detected = MagicMock()

        monitor, _, _ = self._make_monitor(
            check_fn=check_fn,
            on_detected=on_detected,
            poll_interval=0.05,
            cooldown_s=1.0,  # 1 second cooldown
        )
        try:
            monitor.start_watching("test_cam", _make_condition("test"))

            # Wait for first detection
            time.sleep(0.4)
            first_call_count = on_detected.call_count
            assert first_call_count >= 1

            # During cooldown, no new detections should fire
            time.sleep(0.3)
            assert on_detected.call_count == first_call_count

            monitor.stop_all()
        finally:
            self._cleanup()
