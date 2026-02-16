"""Hardware detection and tier auto-selection for Jetson devices."""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class JetsonTier(Enum):
    """Jetson hardware tiers with associated config and compose files."""

    THOR = "thor"
    ORIN = "orin"
    NANO = "nano"

    @property
    def config(self) -> str:
        """Path to the tier-specific YAML configuration file."""
        return f"configs/{self.value}.yaml"

    @property
    def compose(self) -> str:
        """Docker Compose filename for this tier."""
        return f"docker-compose.{self.value}.yml"


def get_vram_gb() -> float:
    """Detect total GPU memory in GB.

    Detection strategy:
      1. Try torch.cuda.get_device_properties(0).total_mem
      2. Fallback: read /proc/meminfo MemTotal (Jetson unified memory)
      3. Return 0.0 if all detection methods fail
    """
    # Strategy 1: PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass

    # Strategy 2: /proc/meminfo (Jetson unified memory)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass

    return 0.0


def detect_tier(vram_gb: float = None) -> JetsonTier:
    """Select hardware tier based on available VRAM.

    Thresholds:
      - >= 96 GB  -> THOR  (Jetson Thor, 128GB unified)
      - >= 16 GB  -> ORIN  (AGX Orin 32-64GB)
      - <  16 GB  -> NANO  (Orin Nano 8GB)

    Args:
        vram_gb: GPU/unified memory in GB. If None, auto-detected via get_vram_gb().

    Returns:
        The appropriate JetsonTier for the detected hardware.
    """
    if vram_gb is None:
        vram_gb = get_vram_gb()

    if vram_gb >= 96:
        tier = JetsonTier.THOR
    elif vram_gb >= 16:
        tier = JetsonTier.ORIN
    else:
        tier = JetsonTier.NANO

    logger.info("Detected %.0fGB VRAM -> %s tier", vram_gb, tier.value)
    return tier
