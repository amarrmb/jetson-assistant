#!/bin/bash
#
# Install specific backends for Jetson Assistant
#
# Usage:
#   ./install_backend.sh qwen     # Install Qwen TTS
#   ./install_backend.sh piper    # Install Piper TTS
#   ./install_backend.sh whisper  # Install Whisper STT
#   ./install_backend.sh all      # Install all backends
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

install_qwen() {
    log_info "Installing Qwen TTS backend..."

    pip install qwen-tts huggingface-hub

    # Check for CUDA and install appropriate PyTorch
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        log_info "CUDA PyTorch already installed"
    else
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi

    log_info "Qwen TTS installed successfully"
}

install_piper() {
    log_info "Installing Piper TTS backend..."

    pip install piper-tts

    log_info "Piper TTS installed successfully"
    log_info "Voice models will be downloaded on first use"
}

install_whisper() {
    log_info "Installing Whisper STT backend..."

    pip install faster-whisper

    log_info "Whisper STT installed successfully"
    log_info "Models will be downloaded on first use"
}

install_sensevoice() {
    log_info "Installing SenseVoice STT backend..."

    pip install funasr modelscope

    log_info "SenseVoice STT installed successfully"
}

install_kokoro() {
    log_info "Installing Kokoro TTS backend..."

    # espeak-ng is a required system dependency
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y espeak-ng
    else
        log_warn "Please install espeak-ng manually (required for Kokoro)"
    fi

    pip install "kokoro>=0.9.4" soundfile

    log_info "Kokoro TTS installed successfully"
    log_info "Model (~164MB) will be downloaded on first use"
}

install_nemotron() {
    log_info "Installing Nemotron Speech STT backend..."

    pip install "nemo_toolkit[asr]>=2.0.0"

    log_info "Nemotron Speech STT installed successfully"
    log_info "Model (~1.2GB) will be downloaded on first use"
}

install_all() {
    log_info "Installing all backends..."
    install_qwen
    install_piper
    install_kokoro
    install_whisper
    install_nemotron
    log_info "All backends installed"
}

show_help() {
    cat << EOF
Install specific backends for Jetson Assistant

Usage: $0 <backend>

Backends:
  qwen        Qwen3-TTS (high-quality neural TTS, requires GPU)
  piper       Piper TTS (lightweight, CPU-friendly)
  kokoro      Kokoro TTS (near-human quality, 82M params, requires espeak-ng)
  whisper     Whisper STT (speech recognition)
  nemotron    Nemotron Speech STT (NVIDIA NeMo, fast English, requires GPU)
  sensevoice  SenseVoice STT (multilingual, good for Asian languages)
  all         Install all backends

Examples:
  $0 qwen      # Install Qwen TTS
  $0 kokoro    # Install Kokoro TTS
  $0 whisper   # Install Whisper STT
  $0 nemotron  # Install Nemotron STT
  $0 all       # Install everything
EOF
}

# Main
case "${1:-}" in
    qwen)
        install_qwen
        ;;
    piper)
        install_piper
        ;;
    whisper)
        install_whisper
        ;;
    kokoro)
        install_kokoro
        ;;
    nemotron)
        install_nemotron
        ;;
    sensevoice)
        install_sensevoice
        ;;
    all)
        install_all
        ;;
    -h|--help|help)
        show_help
        ;;
    "")
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown backend: $1"
        show_help
        exit 1
        ;;
esac
