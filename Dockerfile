# ── Jetson Assistant — Jetson Thor (JetPack 7, CUDA 13, sm_110) ──────────
#
# On-device voice + vision AI. Pre-bundles PyTorch 2.9.1, flash-attn 2.8.3
# (sm_110), Kokoro TTS, Nemotron STT, and all audio dependencies.
# No compilation needed — just pull and run.
#
# Build on Thor:
#   docker build -t jetson-assistant:thor .
#
# Run with docker compose:
#   docker compose up -d       # starts vLLM + assistant
#
# Adapting for other platforms:
#   - Change the base image CUDA version
#   - Change the PyTorch index URL for your CUDA version
#   - Rebuild flash-attn for your GPU arch (see wheels/README.md)
#   - Adjust configs/ for your hardware

FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_COMPILE_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ── System dependencies ──────────────────────────────────────────────────
# Audio: espeak-ng (Kokoro dep), ALSA, portaudio, sndfile, ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    espeak-ng libespeak-ng1 \
    libsndfile1 ffmpeg alsa-utils \
    portaudio19-dev libasound2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. PyTorch from NVIDIA Jetson AI Lab (CUDA 13.0 / aarch64 SBSA) ─────
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.9.1 torchvision==0.24.1 torchaudio \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# ── 2. Pre-built flash-attn (sm_110 — saves ~1hr compile on Thor) ────────
# Build your own: see wheels/README.md
# Triton is needed by flash-attn's rotary embedding ops (used by transformers)
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --break-system-packages \
    triton \
    /tmp/wheels/flash_attn-*.whl \
    && rm -rf /tmp/wheels

# ── 3. Application + recommended backends ─────────────────────────────────
# Note: [assistant] extra includes openwakeword which needs tflite-runtime
# (no aarch64 wheel). The Thor demo uses energy-based wake word detection,
# so we install assistant deps individually, skipping openwakeword.
COPY . .
RUN pip install --no-cache-dir --break-system-packages \
    -e ".[kokoro,nemotron,vision]" \
    sounddevice>=0.4.6 webrtcvad>=2.0.10 ollama>=0.2.0 openai>=1.0 \
    "setuptools<82"

# Remove PEP 668 marker so runtime pip calls (e.g. Kokoro voice download) work
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

EXPOSE 8080 9090

ENTRYPOINT ["jetson-assistant"]
CMD ["assistant", "--config", "configs/thor-sota.yaml"]
