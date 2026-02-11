"""
CameraPool — manages multiple named camera connections (USB + RTSP).

Camera config is loaded from a shared JSON file that can be written by:
- Web UI (console)
- Mobile companion app
- Manual edit
- Voice assistant (add_camera tool as fallback)

Requires: opencv-python-headless >= 4.8.0
    pip install jetson-speech[vision]
"""

import json
import sys
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CONFIG_PATH = Path.home() / ".assistant_cameras.json"


@dataclass
class CameraSource:
    """A named camera source (USB or RTSP)."""

    name: str  # "garage", "front_door", "baby_room", "local"
    url: str  # "rtsp://user:pass@192.0.2.50:554/stream1" or "usb:0"
    width: int = 640
    height: int = 480
    location: str = ""  # Human-readable location hint


class CameraPool:
    """
    Manages multiple named camera connections (USB + RTSP).

    Cameras are lazy-connected: VideoCapture is only opened when a frame
    is first requested, not on startup. This avoids holding N idle RTSP
    connections.

    Usage:
        pool = CameraPool()          # Loads ~/.assistant_cameras.json
        pool.reload()                 # Re-read config file
        cameras = pool.list_cameras() # List all known cameras
        frame_b64 = pool.capture_base64("garage")  # Grab a frame
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._sources: dict[str, CameraSource] = {}
        self._captures: dict = {}  # name -> cv2.VideoCapture
        self._locks: dict[str, threading.Lock] = {}
        self._pool_lock = threading.Lock()  # protects _sources/_captures/_locks dicts
        self._cv2 = None

        self._load_config()

    def _ensure_cv2(self):
        """Lazy-import OpenCV."""
        if self._cv2 is not None:
            return True
        try:
            import cv2
            self._cv2 = cv2
            return True
        except ImportError:
            print(
                "CameraPool: opencv-python-headless not installed. "
                "Install with: pip install opencv-python-headless",
                file=sys.stderr,
            )
            return False

    # ── Config management ──

    def _load_config(self) -> None:
        """Load camera sources from JSON config file."""
        if not self._config_path.exists():
            return

        try:
            data = json.loads(self._config_path.read_text())
            if not isinstance(data, list):
                return
            for entry in data:
                if not isinstance(entry, dict) or "name" not in entry or "url" not in entry:
                    continue
                source = CameraSource(
                    name=entry["name"],
                    url=entry["url"],
                    width=entry.get("width", 640),
                    height=entry.get("height", 480),
                    location=entry.get("location", ""),
                )
                self._sources[source.name] = source
                if source.name not in self._locks:
                    self._locks[source.name] = threading.Lock()
        except (json.JSONDecodeError, OSError) as e:
            print(f"CameraPool: config load error: {e}", file=sys.stderr)

    def _save_config(self) -> None:
        """Persist current sources to JSON config file."""
        data = []
        for source in self._sources.values():
            if source.name == "local":
                continue  # Don't persist the auto-registered local camera
            data.append(asdict(source))

        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            print(f"CameraPool: config save error: {e}", file=sys.stderr)

    def reload(self) -> int:
        """Re-read config file. Returns number of cameras loaded."""
        with self._pool_lock:
            old_names = set(self._sources.keys())
            self._sources.clear()
            self._load_config()

            # Close captures for removed cameras
            removed = old_names - set(self._sources.keys())
            for name in removed:
                self._close_capture(name)

        return len(self._sources)

    # ── Camera management ──

    def add(self, name: str, url: str, width: int = 640, height: int = 480,
            location: str = "") -> str:
        """Add a camera source. Saves to config file."""
        with self._pool_lock:
            source = CameraSource(name=name, url=url, width=width, height=height,
                                  location=location)
            self._sources[name] = source
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            self._save_config()
        return f"Camera '{name}' added ({url})."

    def remove(self, name: str) -> str:
        """Remove a camera source. Closes connection and saves config."""
        with self._pool_lock:
            if name not in self._sources:
                return f"Camera '{name}' not found."
            self._close_capture(name)
            del self._sources[name]
            self._locks.pop(name, None)
            self._save_config()
        return f"Camera '{name}' removed."

    def add_local(self, camera) -> None:
        """Auto-register the local USB camera from the existing Camera instance.

        Args:
            camera: A vision.Camera instance (already opened).
        """
        source = CameraSource(
            name="local",
            url=f"usb:{camera.config.device}",
            width=camera.config.width,
            height=camera.config.height,
            location="Local USB camera",
        )
        with self._pool_lock:
            self._sources["local"] = source
            if "local" not in self._locks:
                self._locks["local"] = threading.Lock()
            # Store the already-open capture directly
            self._captures["local"] = camera._cap

    def add_remote(self, remote_camera) -> None:
        """Auto-register a RemoteCamera (Aether SFU WebRTC stream) as 'phone'.

        Args:
            remote_camera: A remote_camera.RemoteCamera instance (already opened).
        """
        source = CameraSource(
            name="phone",
            url=f"udp:{remote_camera._port}",
            width=remote_camera._width,
            height=remote_camera._height,
            location="Phone camera (Aether WebRTC)",
        )
        with self._pool_lock:
            self._sources["phone"] = source
            if "phone" not in self._locks:
                self._locks["phone"] = threading.Lock()
            # Store the RemoteCamera instance as a sentinel — capture_frame
            # checks for this type and delegates to it directly.
            self._captures["phone"] = remote_camera

    # ── Read-only operations ──

    def list_cameras(self) -> list[CameraSource]:
        """Return list of all known camera sources."""
        with self._pool_lock:
            return list(self._sources.values())

    def has(self, name: str) -> bool:
        """Check if a camera with the given name exists."""
        return name in self._sources

    def capture_frame(self, name: str) -> Optional[np.ndarray]:
        """Capture a single BGR frame from the named camera.

        Opens the connection lazily on first capture. Flushes 3 frames
        to avoid stale RTSP/V4L2 buffers.

        Returns:
            numpy BGR array, or None if capture failed.
        """
        source = self._sources.get(name)
        if source is None:
            return None

        lock = self._locks.get(name)
        if lock is None:
            return None

        with lock:
            # Lazy-open connection
            if name not in self._captures or self._captures[name] is None:
                if not self._ensure_cv2():
                    return None
                if not self._open(name):
                    return None

            cap = self._captures[name]

            # Delegate to RemoteCamera if it's a remote capture
            if hasattr(cap, 'capture_frame') and not hasattr(cap, 'isOpened'):
                return cap.capture_frame()

            if not self._ensure_cv2():
                return None

            # Check if capture is still valid
            if not cap.isOpened():
                if not self._reconnect(name):
                    return None
                cap = self._captures[name]

            # Flush stale frames
            for _ in range(3):
                cap.grab()

            ret, frame = cap.read()
            if not ret or frame is None:
                # Try reconnect once
                if self._reconnect(name):
                    cap = self._captures[name]
                    for _ in range(3):
                        cap.grab()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        return None
                else:
                    return None

            return frame

    def capture_base64(self, name: str, jpeg_quality: int = 85) -> Optional[str]:
        """Capture a frame and return as base64-encoded JPEG.

        Returns:
            Base64 string, or None if capture failed.
        """
        frame = self.capture_frame(name)
        if frame is None:
            return None

        import base64

        cv2 = self._cv2
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        success, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return None

        return base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    # ── Internal helpers ──

    def _parse_url(self, url: str):
        """Parse URL into cv2.VideoCapture argument.

        Returns int for USB devices, string for RTSP/HTTP streams.
        """
        if url.startswith("usb:"):
            return int(url[4:])
        return url

    def _open(self, name: str) -> bool:
        """Open a cv2.VideoCapture for the named camera."""
        cv2 = self._cv2
        source = self._sources.get(name)
        if source is None:
            return False

        try:
            cap_arg = self._parse_url(source.url)

            # Set timeouts for RTSP to avoid long hangs
            if isinstance(cap_arg, str):
                cap = cv2.VideoCapture(cap_arg, cv2.CAP_FFMPEG)
                # RTSP timeout (5 seconds)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            else:
                cap = cv2.VideoCapture(cap_arg)

            if not cap.isOpened():
                print(f"CameraPool: failed to open '{name}' ({source.url})", file=sys.stderr)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, source.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, source.height)

            self._captures[name] = cap
            print(f"CameraPool: opened '{name}' ({source.url})", file=sys.stderr)
            return True

        except Exception as e:
            print(f"CameraPool: error opening '{name}': {e}", file=sys.stderr)
            return False

    def _reconnect(self, name: str) -> bool:
        """Close and reopen a camera connection."""
        self._close_capture(name)
        return self._open(name)

    def _close_capture(self, name: str) -> None:
        """Close the cv2.VideoCapture for a camera if open."""
        cap = self._captures.pop(name, None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def close_all(self) -> None:
        """Close all open camera connections."""
        with self._pool_lock:
            for name in list(self._captures.keys()):
                self._close_capture(name)

    def __len__(self) -> int:
        return len(self._sources)

    def __repr__(self) -> str:
        names = ", ".join(self._sources.keys())
        return f"CameraPool({len(self._sources)} cameras: {names})"
