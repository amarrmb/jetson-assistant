#!/bin/bash
#
# Jetson-specific installation script for Jetson Speech
#
# This script installs dependencies optimized for NVIDIA Jetson devices.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_jetson() {
    if [[ -f /proc/device-tree/compatible ]]; then
        if grep -qi "tegra\|nvidia" /proc/device-tree/compatible 2>/dev/null; then
            return 0
        fi
    fi

    if hostname | grep -qi "jetson"; then
        return 0
    fi

    return 1
}

check_cuda() {
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep -oP 'release \K[0-9.]+' | head -1
        return 0
    fi
    return 1
}

install_uv() {
    if command -v uv &> /dev/null; then
        log_info "uv already installed"
        return 0
    fi

    log_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

install_base() {
    log_info "Installing base dependencies..."

    # System packages
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-dev \
            libsndfile1 \
            ffmpeg \
            alsa-utils \
            espeak-ng
    fi
}

install_pytorch_jetson() {
    log_info "Installing PyTorch for Jetson..."

    # Detect JetPack version
    if [[ -f /etc/nv_tegra_release ]]; then
        JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -oP 'R\K[0-9]+')
        log_info "Detected JetPack major version: $JETPACK_VERSION"
    else
        log_warn "Could not detect JetPack version"
        JETPACK_VERSION="36"  # Default to JetPack 6
    fi

    # Install PyTorch from NVIDIA wheels
    # See: https://developer.nvidia.com/embedded/downloads
    if [[ "$JETPACK_VERSION" -ge "36" ]]; then
        # JetPack 6.x
        pip3 install --no-cache-dir \
            torch torchvision torchaudio \
            --index-url https://developer.download.nvidia.com/compute/redist/jp/v60
    else
        # Older JetPack versions
        log_warn "Older JetPack version detected. Please install PyTorch manually."
        log_warn "See: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-projects/78"
    fi
}

install_jetson_speech() {
    log_info "Installing Jetson Speech..."

    cd "$PROJECT_DIR"

    # Create virtual environment if needed
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate

    # Install base package
    pip install -e .

    # Install backends based on available resources
    log_info "Installing TTS backends..."
    pip install -e ".[qwen]" || log_warn "Qwen TTS installation failed"

    log_info "Installing STT backends..."
    pip install -e ".[whisper]" || log_warn "Whisper installation failed"

    # Optional SOTA backends (Kokoro TTS + Nemotron STT)
    log_info "Installing Kokoro TTS (optional)..."
    pip install -e ".[kokoro]" || log_warn "Kokoro TTS installation failed (install espeak-ng if missing)"

    log_info "Installing Nemotron STT (optional)..."
    pip install -e ".[nemotron]" || log_warn "Nemotron STT installation failed (large dependency)"

    log_info "Installing web UI..."
    pip install -e ".[webui]" || log_warn "Web UI installation failed"
}

setup_swap() {
    # Check if swap is needed for large models
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')

    if [[ "$TOTAL_MEM" -lt 8 ]]; then
        log_warn "Low memory detected (${TOTAL_MEM}GB)"
        log_warn "Consider adding swap for large models:"
        log_warn "  sudo fallocate -l 8G /swapfile"
        log_warn "  sudo chmod 600 /swapfile"
        log_warn "  sudo mkswap /swapfile"
        log_warn "  sudo swapon /swapfile"
    fi
}

main() {
    log_info "=== Jetson Speech Installation ==="

    # Check if running on Jetson
    if check_jetson; then
        log_info "Jetson device detected"
    else
        log_warn "Not running on Jetson device"
        log_warn "This script is optimized for Jetson. Continue anyway? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            log_info "Use 'pip install -e .[all]' for standard installation"
            exit 0
        fi
    fi

    # Check CUDA
    CUDA_VERSION=$(check_cuda)
    if [[ -n "$CUDA_VERSION" ]]; then
        log_info "CUDA $CUDA_VERSION detected"
    else
        log_warn "CUDA not found"
    fi

    # Install components
    install_base
    install_uv
    install_jetson_speech

    # Setup recommendations
    setup_swap

    log_info "=== Installation Complete ==="
    echo ""
    log_info "Activate the environment:"
    log_info "  source .venv/bin/activate"
    echo ""
    log_info "Test TTS:"
    log_info "  jetson-speech tts \"Hello from Jetson\""
    echo ""
    log_info "Start the server:"
    log_info "  jetson-speech serve --port 8080"
}

main "$@"
