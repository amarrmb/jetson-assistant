"""
RemoteCamera — captures frames from Aether SFU WebRTC stream (UDP).

Self-contained RTP/VP8/H.264 receiver that matches the Camera interface
from vision.py. Does NOT depend on LeRobot or the AetherCamera plugin.

The Aether SFU forwards raw RTP packets from a phone's WebRTC stream
to a local UDP port (e.g. 5001). This module listens on that port,
decodes the video, and exposes the latest frame on demand.

Usage:
    cam = RemoteCamera(port=5001)
    if cam.open():
        frame_b64 = cam.capture_base64()
        cam.close()
"""

import logging
import socket
import threading
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np

try:
    import av
    av.logging.set_level(av.logging.ERROR)
    _HAS_AV = True
except ImportError:
    _HAS_AV = False


class _RTPVideoDecoder:
    """Decode VP8 or H.264 RTP packets into numpy frames using PyAV.

    Auto-detects the codec from the RTP payload structure:
    - VP8 (RFC 7741): payload descriptor with X/S/PartID bits
    - H.264 (RFC 6184): NAL unit types (single, STAP-A, FU-A)

    Adapted from lerobot_camera_aether/camera_aether.py (self-contained,
    no LeRobot dependency).
    """

    def __init__(self) -> None:
        self._codec_name: Optional[str] = None
        self._codec: object = None  # av.CodecContext
        self._nals = bytearray()
        self._fua_buf = bytearray()
        self._vp8_buf = bytearray()
        self._got_keyframe = False

    def reset(self) -> None:
        self._nals.clear()
        self._fua_buf.clear()
        self._vp8_buf.clear()
        self._got_keyframe = False

    def _ensure_codec(self, name: str) -> None:
        if self._codec_name != name:
            self._codec_name = name
            self._codec = av.CodecContext.create(name, "r")

    def feed(self, rtp_data: bytes) -> Optional[np.ndarray]:
        """Feed a raw RTP packet. Returns a decoded BGR frame or None."""
        if len(rtp_data) < 13:
            return None

        marker = bool(rtp_data[1] & 0x80)
        payload = rtp_data[12:]

        nal_type = payload[0] & 0x1F
        if nal_type in (24, 28) or (self._codec_name == "h264"):
            return self._feed_h264(payload, marker)
        else:
            return self._feed_vp8(payload, marker)

    # -- VP8 (RFC 7741) ---------------------------------------------------

    @staticmethod
    def _strip_vp8_descriptor(payload: bytes) -> tuple:
        if not payload:
            return b"", False, False
        idx = 0
        b0 = payload[idx]
        x = bool(b0 & 0x80)
        s = bool(b0 & 0x10)
        idx += 1

        if x and idx < len(payload):
            ext = payload[idx]
            idx += 1
            if ext & 0x80 and idx < len(payload):
                if payload[idx] & 0x80:
                    idx += 2
                else:
                    idx += 1
            if ext & 0x40 and idx < len(payload):
                idx += 1
            if (ext & 0x20 or ext & 0x10) and idx < len(payload):
                idx += 1

        vp8_data = payload[idx:]
        is_keyframe = False
        if s and len(vp8_data) > 0:
            is_keyframe = not bool(vp8_data[0] & 0x01)
        return vp8_data, s, is_keyframe

    def _feed_vp8(self, payload: bytes, marker: bool) -> Optional[np.ndarray]:
        self._ensure_codec("vp8")
        vp8_data, is_start, is_keyframe = self._strip_vp8_descriptor(payload)

        if not self._got_keyframe:
            if is_start and is_keyframe:
                self._got_keyframe = True
                self._vp8_buf.clear()
            else:
                return None

        if is_start:
            self._vp8_buf.clear()
        self._vp8_buf.extend(vp8_data)

        if not marker:
            return None

        if not self._vp8_buf:
            return None

        packet = av.Packet(bytes(self._vp8_buf))
        self._vp8_buf.clear()
        try:
            frames = self._codec.decode(packet)
            if frames:
                return frames[-1].to_ndarray(format="bgr24")
        except av.error.InvalidDataError:
            self._got_keyframe = False
        return None

    # -- H.264 (RFC 6184) -------------------------------------------------

    def _feed_h264(self, payload: bytes, marker: bool) -> Optional[np.ndarray]:
        self._ensure_codec("h264")
        nal_type = payload[0] & 0x1F

        if 1 <= nal_type <= 23:
            self._nals.extend(b'\x00\x00\x00\x01')
            self._nals.extend(payload)
        elif nal_type == 24:
            i = 1
            while i + 2 <= len(payload):
                size = (payload[i] << 8) | payload[i + 1]
                i += 2
                if i + size <= len(payload):
                    self._nals.extend(b'\x00\x00\x00\x01')
                    self._nals.extend(payload[i:i + size])
                i += size
        elif nal_type == 28:
            if len(payload) < 2:
                return None
            fu_indicator = payload[0]
            fu_header = payload[1]
            start = bool(fu_header & 0x80)
            end = bool(fu_header & 0x40)
            nal_hdr = (fu_indicator & 0xE0) | (fu_header & 0x1F)
            if start:
                self._fua_buf.clear()
                self._fua_buf.append(nal_hdr)
            self._fua_buf.extend(payload[2:])
            if end:
                self._nals.extend(b'\x00\x00\x00\x01')
                self._nals.extend(self._fua_buf)
                self._fua_buf.clear()

        if not marker:
            return None

        if not self._nals:
            return None

        packet = av.Packet(bytes(self._nals))
        self._nals.clear()
        try:
            frames = self._codec.decode(packet)
            if frames:
                return frames[-1].to_ndarray(format="bgr24")
        except av.error.InvalidDataError:
            pass
        return None


class RemoteCamera:
    """Captures frames from Aether SFU WebRTC stream (UDP).

    Listens for raw RTP packets on a UDP port, decodes VP8/H.264
    via PyAV, and stores the latest frame for on-demand capture.

    Matches the Camera interface from vision.py (is_open, open, close,
    capture_frame, capture_base64).
    """

    def __init__(self, port: int = 5000, width: int = 640, height: int = 480):
        self._port = port
        self._width = width
        self._height = height
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_open(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def open(self) -> bool:
        """Start UDP listener thread, return True on success."""
        if not _HAS_AV:
            logger.error("RemoteCamera: PyAV not installed. Install with: pip install av")
            return False

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("127.0.0.1", self._port))
            self._sock.settimeout(1.0)
        except OSError as e:
            logger.error("RemoteCamera: bind error on port %d: %s", self._port, e)
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name=f"remote-cam-{self._port}"
        )
        self._thread.start()
        return True

    def close(self) -> None:
        """Stop listener, release resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        with self._frame_lock:
            self._frame = None

    def capture_frame(self, flush: bool = True) -> Optional[np.ndarray]:
        """Return latest decoded BGR frame or None."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def capture_base64(self) -> Optional[str]:
        """Return latest frame as base64 JPEG or None."""
        frame = self.capture_frame()
        if frame is None:
            return None

        import base64
        import cv2

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return None

        return base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    def _receive_loop(self) -> None:
        """Background thread: receive RTP packets and decode frames."""
        decoder = _RTPVideoDecoder()

        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            frame = decoder.feed(data)
            if frame is not None:
                with self._frame_lock:
                    self._frame = frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
