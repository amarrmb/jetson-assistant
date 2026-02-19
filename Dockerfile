# ── Jetson Assistant — Multi-Platform ──────────────────────────────────
#
# On-device voice + vision AI. Pre-bundles PyTorch, Kokoro TTS, Nemotron
# STT, and all audio dependencies.
#
# Build args select the platform:
#   PLATFORM=thor  (default) — JetPack 7, CUDA 13.0, sm_110 (SBSA)
#   PLATFORM=orin  — JetPack 6.x, CUDA 12.6, sm_87 (tegra)
#   PLATFORM=nano  — JetPack 6.x, CUDA 12.6, sm_87 (tegra, no VLM)
#   PLATFORM=spark — DGX Spark, CUDA 13.0, sm_121 (Blackwell GB10)
#
# Build:
#   docker build -t jetson-assistant:thor .                            # Thor (default)
#   docker build -t jetson-assistant:orin --build-arg PLATFORM=orin .  # AGX Orin
#   docker build -t jetson-assistant:nano --build-arg PLATFORM=nano .  # Orin Nano
#   docker build -t jetson-assistant:spark --build-arg PLATFORM=spark . # DGX Spark
#
# Run with docker compose:
#   docker compose -f docker-compose.thor.yml up -d
#   docker compose -f docker-compose.orin.yml up -d
#   docker compose -f docker-compose.nano.yml up -d
#   docker compose -f docker-compose.spark.yml up -d

# ── Platform selection ───────────────────────────────────────────────
ARG PLATFORM=thor

# Per-platform base images (BuildKit only pulls the selected one)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 AS base-thor
FROM nvcr.io/nvidia/l4t-cuda:12.6.68-runtime AS base-orin
FROM nvcr.io/nvidia/l4t-cuda:12.6.68-runtime AS base-nano
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 AS base-spark
FROM base-${PLATFORM} AS base

# Re-declare after FROM (Docker resets ARGs)
ARG PLATFORM=thor

# ── Platform-specific build args ─────────────────────────────────────

# Thor: JetPack 7, CUDA 13.0, Ubuntu 24.04 (SBSA architecture)
# Pin torch 2.9.1 + torchaudio 2.9.1 — flash-attn wheel requires 2.9.x ABI.
# No torchvision — SBSA index only has 0.25.0 (needs torch 2.10), not used by us.
ARG PYTORCH_INDEX_thor=https://pypi.jetson-ai-lab.io/sbsa/cu130
ARG PYTORCH_PKGS_thor="torch==2.9.1 torchaudio==2.9.1"
ARG INSTALL_FLASH_ATTN_thor=true
ARG DEFAULT_CONFIG_thor=configs/thor-sota.yaml
ARG EXTRAS_thor=kokoro,nemotron,vision,search

# Orin: JetPack 6.x, CUDA 12.6, Ubuntu 22.04 (tegra architecture)
ARG PYTORCH_INDEX_orin=https://pypi.jetson-ai-lab.io/jp6/cu126
ARG PYTORCH_PKGS_orin="torch torchvision torchaudio"
ARG INSTALL_FLASH_ATTN_orin=false
ARG DEFAULT_CONFIG_orin=configs/orin.yaml
ARG EXTRAS_orin=kokoro,nemotron,vision,search

# Nano: Same base as Orin but lighter config (no VLM, uses Ollama + Piper)
ARG PYTORCH_INDEX_nano=https://pypi.jetson-ai-lab.io/jp6/cu126
ARG PYTORCH_PKGS_nano="torch torchaudio"
ARG INSTALL_FLASH_ATTN_nano=false
ARG DEFAULT_CONFIG_nano=configs/nano.yaml
ARG EXTRAS_nano=whisper,piper,search

# DGX Spark: Blackwell GB10, CUDA 13.0, Ubuntu 24.04 (SBSA wheels — same as Thor)
# Same SBSA constraint as Thor: pin to 2.9.1, no torchvision.
ARG PYTORCH_INDEX_spark=https://pypi.jetson-ai-lab.io/sbsa/cu130
ARG PYTORCH_PKGS_spark="torch==2.9.1 torchaudio==2.9.1"
ARG INSTALL_FLASH_ATTN_spark=false
ARG DEFAULT_CONFIG_spark=configs/spark.yaml
ARG EXTRAS_spark=kokoro,nemotron,vision,search

LABEL org.opencontainers.image.source="https://github.com/amarrmb/jetson-assistant" \
      org.opencontainers.image.description="On-device voice + vision AI for NVIDIA Jetson" \
      org.opencontainers.image.license="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_COMPILE_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ── System dependencies ──────────────────────────────────────────────
# Audio: espeak-ng (Kokoro dep), ALSA, portaudio, sndfile, ffmpeg
# pipewire-alsa: ALSA→PipeWire plugin for Bluetooth and system default support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    espeak-ng libespeak-ng1 \
    libsndfile1 ffmpeg alsa-utils \
    portaudio19-dev libasound2-dev \
    pipewire-alsa \
    curl \
    # NVPL + cuDSS: required by Jetson AI Lab SBSA PyTorch wheels (Thor + Spark)
    libnvpl-lapack0 libnvpl-blas0 libcudss0-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

# Add non-standard NVIDIA lib paths for SBSA wheels (libcudss, libnvpl)
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/13:${LD_LIBRARY_PATH}"

WORKDIR /app

# ── 1. PyTorch from NVIDIA Jetson AI Lab ─────────────────────────────
# Platform selects the right index URL and packages via build args.
RUN PYTORCH_INDEX=$(eval echo \${PYTORCH_INDEX_${PLATFORM}}) && \
    PYTORCH_PKGS=$(eval echo \${PYTORCH_PKGS_${PLATFORM}}) && \
    echo "Installing PyTorch for ${PLATFORM}: ${PYTORCH_PKGS} from ${PYTORCH_INDEX}" && \
    pip install --no-cache-dir --break-system-packages \
        ${PYTORCH_PKGS} \
        --index-url ${PYTORCH_INDEX}

# SBSA PyTorch wheels (Thor/Spark) link against libcupti but the runtime
# base image doesn't include it. Install from CUDA apt repo.
# L4T base images (Orin/Nano) already have CUPTI via JetPack.
RUN if [ "${PLATFORM}" = "thor" ] || [ "${PLATFORM}" = "spark" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends cuda-cupti-13-0 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ── 2. Flash-attn (Thor only — pre-built wheel for sm_110) ──────────
# Orin/Nano skip this step — they use SDPA attention instead.
# To add flash-attn for Orin, build a wheel for sm_87 (see wheels/README.md).
COPY wheels/ /tmp/wheels/
RUN INSTALL_FA=$(eval echo \${INSTALL_FLASH_ATTN_${PLATFORM}}) && \
    if [ "${INSTALL_FA}" = "true" ]; then \
        echo "Installing flash-attn for ${PLATFORM}" && \
        pip install --no-cache-dir --break-system-packages \
            triton \
            /tmp/wheels/flash_attn-*.whl; \
    else \
        echo "Skipping flash-attn for ${PLATFORM} (uses SDPA)"; \
    fi && \
    rm -rf /tmp/wheels

# ── 3. Application + recommended backends ────────────────────────────
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
COPY . .
RUN EXTRAS=$(eval echo \${EXTRAS_${PLATFORM}}) && \
    echo "Installing extras: ${EXTRAS}" && \
    pip install --no-cache-dir --break-system-packages \
        -e ".[${EXTRAS}]" \
        sounddevice>=0.4.6 webrtcvad>=2.0.10 ollama>=0.2.0 openai>=1.0 \
        aiohttp>=3.9 cryptography>=41.0 "duckduckgo-search>=7.0" \
        "setuptools<82"

# Pre-download spacy model (Kokoro TTS dependency, ~13MB)
# Use pip directly — `spacy download` imports torch which needs NVPL (runtime only)
RUN pip install --no-cache-dir --break-system-packages \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Remove PEP 668 marker so runtime pip calls (e.g. Kokoro voice download) work
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

EXPOSE 8080 9090

ENTRYPOINT ["docker-entrypoint.sh"]
# Write platform-specific default config path (read by entrypoint when no args)
RUN DEFAULT_CFG=$(eval echo \${DEFAULT_CONFIG_${PLATFORM}}) && \
    echo "${DEFAULT_CFG}" > /app/.default-config
# Compose files override CMD; without compose, entrypoint reads .default-config
