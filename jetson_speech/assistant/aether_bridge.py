"""
AetherBridge — connects voice assistant to Aether Hub for alerts and remote commands.

Graceful degradation: if aether-sdk-python is not installed or Hub is unavailable,
AetherBridge is None and everything still works — just no alerts pushed to Hub.
"""

import sys
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger("AetherBridge")


class AetherBridge:
    """
    Connects voice assistant to Aether Hub for alerts and remote commands.

    Usage:
        bridge = AetherBridge.create("localhost", port=8000, pin="your-pin")
        if bridge:
            bridge.publish_alert("garage", "door opened")
    """

    def __init__(self, client, hub_host: str, hub_port: int):
        self._client = client
        self._hub_host = hub_host
        self._hub_port = hub_port

    @staticmethod
    def create(
        hub_host: str,
        hub_port: int = 8000,
        pin: str = "",
        name: str = "home-assistant",
    ) -> Optional["AetherBridge"]:
        """Create and connect an AetherBridge. Returns None on failure."""
        try:
            from aether_sdk.client import AetherClient
        except ImportError:
            print(
                "AetherBridge: aether-sdk not installed. "
                "Alerts will not be pushed to Hub.",
                file=sys.stderr,
            )
            return None

        try:
            client = AetherClient(
                host=hub_host,
                port=hub_port,
                pin=pin,
                name=name,
                capabilities=["camera_alerts", "voice_assistant"],
                auto_reconnect=True,
            )
            client.connect()

            # Wait briefly for connection
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not client.is_authenticated:
                time.sleep(0.2)

            if client.is_authenticated:
                print(
                    f"AetherBridge: connected to Hub at {hub_host}:{hub_port}",
                    file=sys.stderr,
                )
                return AetherBridge(client, hub_host, hub_port)

            print(
                f"AetherBridge: failed to authenticate with Hub at {hub_host}:{hub_port}",
                file=sys.stderr,
            )
            client.disconnect()
            return None

        except Exception as e:
            print(f"AetherBridge: connection error: {e}", file=sys.stderr)
            return None

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected and self._client.is_authenticated

    def publish_alert(self, camera_name: str, condition: str) -> bool:
        """Broadcast camera alert to all Hub clients."""
        return self._client.send({
            "type": "CAMERA_ALERT",
            "camera": camera_name,
            "condition": condition,
            "timestamp": time.time(),
            "source": "voice-assistant",
        })

    def publish_status(self, cameras: list[str], watches: list[dict]) -> bool:
        """Publish current assistant status for discovery."""
        return self._client.send({
            "type": "ASSISTANT_STATUS",
            "cameras": cameras,
            "watches": watches,
            "timestamp": time.time(),
        })

    def send_to(self, target_id: str, payload: dict) -> bool:
        """Send a directed message to a specific client via Hub."""
        return self._client.send({
            "type": "SEND_TO",
            "to": target_id,
            "payload": payload,
        })

    def on_command(self, handler: Callable[[dict], None]) -> None:
        """Register handler for incoming commands from mobile/web.

        Hub wraps SEND_TO as DIRECT_MESSAGE when delivering to us,
        so we must handle that type as well.
        """
        def _message_handler(data: dict):
            msg_type = data.get("type", "")
            if msg_type in ("ASSISTANT_CMD", "SEND_TO", "DIRECT_MESSAGE"):
                handler(data)

        self._client.on_message = _message_handler

    def disconnect(self) -> None:
        """Disconnect from Hub."""
        if self._client:
            self._client.disconnect()
            print("AetherBridge: disconnected", file=sys.stderr)
