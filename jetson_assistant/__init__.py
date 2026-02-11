"""
Jetson Assistant - Modular TTS + STT server for edge devices.
"""

import os
import warnings

# Suppress verbose warnings for cleaner output
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress onnxruntime and transformers warnings
import logging
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

__version__ = "0.1.0"

from jetson_assistant.core.engine import Engine

__all__ = ["Engine", "__version__"]
