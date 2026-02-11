"""
Tests for CameraPool multi-camera management.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCameraPool:
    """Tests for CameraPool camera management."""

    def _make_pool(self, cameras=None):
        """Create a CameraPool with a temp config file."""
        from jetson_assistant.assistant.cameras import CameraPool

        tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        if cameras:
            json.dump(cameras, tmpfile)
        else:
            json.dump([], tmpfile)
        tmpfile.close()

        pool = CameraPool(config_path=Path(tmpfile.name))
        self._tmpfile = tmpfile.name
        return pool

    def _cleanup(self):
        import os
        if hasattr(self, "_tmpfile") and os.path.exists(self._tmpfile):
            os.unlink(self._tmpfile)

    def test_empty_pool(self):
        """Empty config creates empty pool."""
        pool = self._make_pool()
        try:
            assert len(pool) == 0
            assert pool.list_cameras() == []
        finally:
            self._cleanup()

    def test_load_config(self):
        """Cameras loaded from config file on init."""
        cameras = [
            {"name": "garage", "url": "rtsp://192.0.2.50:554/stream1", "location": "Garage"},
            {"name": "front", "url": "rtsp://192.0.2.51:554/stream1"},
        ]
        pool = self._make_pool(cameras)
        try:
            assert len(pool) == 2
            assert pool.has("garage")
            assert pool.has("front")
            assert not pool.has("nonexistent")
        finally:
            self._cleanup()

    def test_add_and_list(self):
        """Add cameras and verify they appear in list."""
        pool = self._make_pool()
        try:
            pool.add("test1", "rtsp://1.2.3.4:554/stream")
            pool.add("test2", "usb:0", location="Desk")

            cameras = pool.list_cameras()
            assert len(cameras) == 2
            names = [c.name for c in cameras]
            assert "test1" in names
            assert "test2" in names

            # Verify location
            test2 = [c for c in cameras if c.name == "test2"][0]
            assert test2.location == "Desk"
        finally:
            self._cleanup()

    def test_remove(self):
        """Add then remove, verify gone."""
        pool = self._make_pool()
        try:
            pool.add("test1", "rtsp://1.2.3.4:554/stream")
            assert pool.has("test1")

            result = pool.remove("test1")
            assert "removed" in result.lower()
            assert not pool.has("test1")
            assert len(pool) == 0
        finally:
            self._cleanup()

    def test_remove_nonexistent(self):
        """Removing a non-existent camera returns error message."""
        pool = self._make_pool()
        try:
            result = pool.remove("ghost")
            assert "not found" in result.lower()
        finally:
            self._cleanup()

    def test_persistence(self):
        """Add cameras, create new pool from same path, verify loaded."""
        from jetson_assistant.assistant.cameras import CameraPool

        pool = self._make_pool()
        try:
            pool.add("cam1", "rtsp://1.2.3.4/stream", location="Room 1")
            pool.add("cam2", "usb:2")

            # Create a new pool from the same config file
            pool2 = CameraPool(config_path=Path(self._tmpfile))
            assert len(pool2) == 2
            assert pool2.has("cam1")
            assert pool2.has("cam2")

            cam1 = [c for c in pool2.list_cameras() if c.name == "cam1"][0]
            assert cam1.location == "Room 1"
        finally:
            self._cleanup()

    def test_usb_url_parsing(self):
        """usb:0 -> int(0), rtsp://... -> string URL."""
        pool = self._make_pool()
        try:
            assert pool._parse_url("usb:0") == 0
            assert pool._parse_url("usb:2") == 2
            assert pool._parse_url("rtsp://1.2.3.4:554/stream") == "rtsp://1.2.3.4:554/stream"
            assert pool._parse_url("http://1.2.3.4/mjpeg") == "http://1.2.3.4/mjpeg"
        finally:
            self._cleanup()

    def test_capture_missing_camera(self):
        """capture_frame returns None for unknown camera name."""
        pool = self._make_pool()
        try:
            assert pool.capture_frame("nonexistent") is None
            assert pool.capture_base64("nonexistent") is None
        finally:
            self._cleanup()

    def test_reload(self):
        """Reload picks up changes to the config file."""
        pool = self._make_pool([
            {"name": "cam1", "url": "usb:0"},
        ])
        try:
            assert len(pool) == 1

            # Modify config file externally
            new_config = [
                {"name": "cam1", "url": "usb:0"},
                {"name": "cam2", "url": "rtsp://1.2.3.4/stream"},
            ]
            Path(self._tmpfile).write_text(json.dumps(new_config))

            count = pool.reload()
            assert count == 2
            assert pool.has("cam2")
        finally:
            self._cleanup()

    def test_local_camera_not_persisted(self):
        """The auto-registered 'local' camera should not be saved to config."""
        pool = self._make_pool()
        try:
            # Simulate adding a local camera via add_local
            mock_camera = MagicMock()
            mock_camera.config.device = 0
            mock_camera.config.width = 640
            mock_camera.config.height = 480
            mock_camera._cap = MagicMock()

            pool.add_local(mock_camera)
            assert pool.has("local")

            # Add a regular camera too
            pool.add("test", "rtsp://1.2.3.4/stream")

            # Read the config file — should only contain "test", not "local"
            config = json.loads(Path(self._tmpfile).read_text())
            names = [c["name"] for c in config]
            assert "test" in names
            assert "local" not in names
        finally:
            self._cleanup()

    def test_repr(self):
        """repr shows camera count and names."""
        pool = self._make_pool([
            {"name": "a", "url": "usb:0"},
            {"name": "b", "url": "usb:1"},
        ])
        try:
            r = repr(pool)
            assert "2 cameras" in r
            assert "a" in r
            assert "b" in r
        finally:
            self._cleanup()

    def test_invalid_config_entries_skipped(self):
        """Config entries missing name or url are silently skipped."""
        pool = self._make_pool([
            {"name": "valid", "url": "usb:0"},
            {"url": "usb:1"},  # missing name
            {"name": "nourl"},  # missing url
            "not_a_dict",  # not a dict
        ])
        try:
            assert len(pool) == 1
            assert pool.has("valid")
        finally:
            self._cleanup()
