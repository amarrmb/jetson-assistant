#!/bin/bash
#
# Qwen3-TTS - Text-to-Speech Helper
# Single script for setup and usage on Jetson/CUDA devices
#
# Usage:
#   ./qwen-tts.sh setup                    # One-time installation
#   ./qwen-tts.sh "Hello world"            # Quick TTS
#   ./qwen-tts.sh -f document.pdf          # Read PDF
#   ./qwen-tts.sh -f book.txt -o audio.wav # Convert file to audio
#   ./qwen-tts.sh --list-speakers          # Show available voices
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.qwen3-tts"
UV_BIN="$HOME/.local/bin/uv"
ENGINE_PY="$SCRIPT_DIR/.tts_engine.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_setup() {
    [[ -f "$UV_BIN" ]] || return 1
    [[ -d "$VENV_DIR" ]] || return 1
    [[ -f "$VENV_DIR/bin/python" ]] || return 1
    [[ -f "$ENGINE_PY" ]] || return 1
    # Check torch with CUDA and qwen_tts
    "$VENV_DIR/bin/python" -c "
import torch
import qwen_tts
assert 'cu130' in torch.__version__ or '+cu130' in torch.__version__
" 2>/dev/null || return 1
    return 0
}

install_uv() {
    if [[ -f "$UV_BIN" ]]; then
        log_info "uv already installed"
        return 0
    fi
    log_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
}

create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment exists"
        return 0
    fi
    log_info "Creating virtual environment (Python 3.12)..."
    "$UV_BIN" venv "$VENV_DIR" --python 3.12
}

check_deps() {
    # Check if key packages are installed with correct versions
    "$VENV_DIR/bin/python" -c "
import torch
import qwen_tts
# Check CUDA torch version
assert 'cu130' in torch.__version__ or '+cu130' in torch.__version__, 'Wrong torch version'
" 2>/dev/null
}

install_deps() {
    if check_deps 2>/dev/null; then
        log_info "Dependencies already installed"
        return 0
    fi

    log_info "Installing qwen-tts..."
    "$UV_BIN" pip install -U qwen-tts -p "$VENV_DIR"

    log_info "Installing PyTorch with CUDA 13.0..."
    "$UV_BIN" pip install --force-reinstall \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu130 \
        -p "$VENV_DIR"

    log_info "Installing additional dependencies..."
    "$UV_BIN" pip install wheel packaging ninja pypdf docx2txt -p "$VENV_DIR"
}

create_engine() {
    if [[ -f "$ENGINE_PY" ]]; then
        log_info "TTS engine already exists"
        return 0
    fi
    log_info "Creating TTS engine..."
    cat > "$ENGINE_PY" << 'PYEOF'
#!/usr/bin/env python3
"""
Qwen3-TTS Engine
Handles text extraction, speech generation, and audio output
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import queue
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from huggingface_hub import snapshot_download

# Speakers and languages
SPEAKERS = {
    "aiden": "Male, young, energetic",
    "dylan": "Male, casual",
    "eric": "Male, professional",
    "ono_anna": "Female, Japanese accent",
    "ryan": "Male, neutral (default)",
    "serena": "Female, warm",
    "sohee": "Female, Korean accent",
    "uncle_fu": "Male, Chinese accent, mature",
    "vivian": "Female, clear",
}

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean",
             "French", "German", "Spanish", "Portuguese", "Russian"]

_model = None
_model_size = None

def get_model(model_size="0.6B"):
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        return _model
    print(f"Loading Qwen3-TTS ({model_size})...", file=sys.stderr)
    model_path = snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice")
    from qwen_tts import Qwen3TTSModel
    _model = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16,
    )
    _model_size = model_size
    print("Model loaded!", file=sys.stderr)
    return _model

def extract_text_from_file(filepath):
    """Extract text from various file formats."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == '.txt':
        return path.read_text(encoding='utf-8')

    elif ext == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except ImportError:
            print("Error: pypdf not installed. Run setup again.", file=sys.stderr)
            sys.exit(1)

    elif ext == '.docx':
        try:
            import docx2txt
            return docx2txt.process(filepath)
        except ImportError:
            print("Error: docx2txt not installed. Run setup again.", file=sys.stderr)
            sys.exit(1)

    elif ext == '.md':
        # Remove markdown formatting
        text = path.read_text(encoding='utf-8')
        text = re.sub(r'#{1,6}\s*', '', text)  # headers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # italic
        text = re.sub(r'`(.+?)`', r'\1', text)  # code
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
        return text

    else:
        # Try reading as plain text
        return path.read_text(encoding='utf-8')

def split_into_chunks(text, max_chars=500, by_sentence=False):
    """Split text into speakable chunks."""
    # Normalize: add space after sentence-ending punctuation if missing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # First split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)

        if by_sentence:
            # For streaming: each sentence is a chunk
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence)
        else:
            # For normal mode: combine sentences up to max_chars
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)

    return chunks

def generate_speech(text, speaker="ryan", language="English", model_size="0.6B",
                   temperature=1.0, top_p=0.9, repetition_penalty=1.0):
    """Generate speech from text."""
    model = get_model(model_size)

    kwargs = {
        "text": text,
        "language": language,
        "speaker": speaker.lower(),
        "non_streaming_mode": True,
        "max_new_tokens": 2048,
    }

    # Add tuning parameters if not default
    if temperature != 1.0:
        kwargs["temperature"] = temperature
    if top_p != 0.9:
        kwargs["top_p"] = top_p
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty

    wavs, sr = model.generate_custom_voice(**kwargs)
    wav_int16 = np.clip(wavs[0] * 32767, -32768, 32767).astype(np.int16)
    return wav_int16, sr

def save_audio(wav, sr, output_path):
    """Save audio to file."""
    wavfile.write(output_path, sr, wav)
    print(f"Saved: {output_path}", file=sys.stderr)

def play_audio(wav, sr):
    """Play audio using aplay."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sr, wav)
        try:
            subprocess.run(["aplay", "-q", f.name], check=True)
        finally:
            os.unlink(f.name)

def concatenate_wavs(wav_list, sr):
    """Concatenate multiple audio arrays."""
    # Add small silence between chunks
    silence = np.zeros(int(sr * 0.3), dtype=np.int16)
    result = []
    for i, wav in enumerate(wav_list):
        result.append(wav)
        if i < len(wav_list) - 1:
            result.append(silence)
    return np.concatenate(result)

def list_speakers():
    """Print available speakers."""
    print("\nAvailable Speakers:")
    print("-" * 40)
    for name, desc in SPEAKERS.items():
        print(f"  {name:12} - {desc}")
    print()

def list_languages():
    """Print available languages."""
    print("\nAvailable Languages:")
    print("-" * 40)
    for lang in LANGUAGES:
        print(f"  {lang}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: Text-to-Speech Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world"                    # Quick TTS
  %(prog)s -f document.pdf -o audio.wav     # Convert PDF to audio
  %(prog)s "Text" -s serena --stream        # Stream with female voice
  %(prog)s -f book.txt --stream             # Stream long document
  %(prog)s --list-speakers                  # Show available voices
        """
    )

    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("-f", "--file", help="Input file (txt, pdf, docx, md)")
    parser.add_argument("-o", "--output", help="Output WAV file")
    parser.add_argument("-s", "--speaker", default="ryan",
                        choices=list(SPEAKERS.keys()), metavar="SPEAKER",
                        help="Speaker voice (default: ryan)")
    parser.add_argument("-l", "--language", default="English",
                        choices=LANGUAGES, metavar="LANG",
                        help="Language (default: English)")
    parser.add_argument("-m", "--model", default="0.6B", choices=["0.6B", "1.7B"],
                        help="Model size (default: 0.6B)")

    # Audio tuning
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0, higher=more varied)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (default: 1.0)")

    # Mode options
    parser.add_argument("--stream", action="store_true",
                        help="Stream: play chunks as generated")
    parser.add_argument("--play", action="store_true",
                        help="Play audio after saving")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress messages")

    # Info options
    parser.add_argument("--list-speakers", action="store_true",
                        help="List available speakers")
    parser.add_argument("--list-languages", action="store_true",
                        help="List available languages")

    args = parser.parse_args()

    # Info commands
    if args.list_speakers:
        list_speakers()
        return
    if args.list_languages:
        list_languages()
        return

    # Get text
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = extract_text_from_file(args.file)
        if not args.quiet:
            print(f"Extracted {len(text)} characters from {args.file}", file=sys.stderr)
    elif args.text:
        text = args.text
    else:
        parser.error("Either text or --file is required")

    text = text.strip()
    if not text:
        print("Error: No text to process", file=sys.stderr)
        sys.exit(1)

    # Process
    if args.stream:
        # Pipelined streaming: generate ahead while playing
        chunks = split_into_chunks(text, by_sentence=True)
        if not args.quiet:
            print(f"Streaming {len(chunks)} chunk(s) (pipelined)...", file=sys.stderr)

        # Pre-warm the model before starting the pipeline
        _ = get_model(args.model)

        # Queue for passing generated audio from producer to consumer
        audio_queue = queue.Queue(maxsize=3)  # Buffer up to 3 chunks ahead
        all_wavs = []
        sample_rate = [None]  # Use list to share across threads
        gen_error = [None]  # Capture any generation errors

        def producer():
            """Generate audio chunks and put them in the queue."""
            try:
                for i, chunk in enumerate(chunks):
                    if not args.quiet:
                        preview = chunk[:60] + ('...' if len(chunk) > 60 else '')
                        print(f"[Gen {i+1}/{len(chunks)}] {preview}", file=sys.stderr)

                    wav, sr = generate_speech(
                        chunk, args.speaker, args.language, args.model,
                        args.temperature, args.top_p, args.repetition_penalty
                    )
                    sample_rate[0] = sr
                    audio_queue.put((i, wav))
            except Exception as e:
                gen_error[0] = e
            finally:
                audio_queue.put(None)  # Signal completion

        def consumer():
            """Play audio chunks as they become available."""
            while True:
                item = audio_queue.get()
                if item is None:
                    break
                i, wav = item
                all_wavs.append((i, wav))
                if not args.quiet:
                    print(f"[Play {i+1}/{len(chunks)}]", file=sys.stderr)
                play_audio(wav, sample_rate[0])

        # Start producer thread (generates audio ahead)
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Consumer runs in main thread
        consumer()

        # Wait for producer to finish
        producer_thread.join()

        if gen_error[0]:
            print(f"Error during generation: {gen_error[0]}", file=sys.stderr)
            sys.exit(1)

        # Also save concatenated if output specified
        if args.output and all_wavs:
            # Sort by index to ensure correct order
            all_wavs.sort(key=lambda x: x[0])
            wavs_only = [w for _, w in all_wavs]
            combined = concatenate_wavs(wavs_only, sample_rate[0])
            save_audio(combined, sample_rate[0], args.output)

    else:
        # Normal mode: generate full audio
        # For long text, split into chunks and concatenate
        chunks = split_into_chunks(text)

        if len(chunks) == 1:
            if not args.quiet:
                preview = text[:80] + ('...' if len(text) > 80 else '')
                print(f"Generating: {preview}", file=sys.stderr)
            wav, sr = generate_speech(
                text, args.speaker, args.language, args.model,
                args.temperature, args.top_p, args.repetition_penalty
            )
        else:
            if not args.quiet:
                print(f"Processing {len(chunks)} chunks...", file=sys.stderr)
            wavs = []
            for i, chunk in enumerate(chunks):
                if not args.quiet:
                    print(f"[{i+1}/{len(chunks)}]", file=sys.stderr)
                w, sr = generate_speech(
                    chunk, args.speaker, args.language, args.model,
                    args.temperature, args.top_p, args.repetition_penalty
                )
                wavs.append(w)
            wav = concatenate_wavs(wavs, sr)

        if args.output:
            save_audio(wav, sr, args.output)
            if args.play:
                play_audio(wav, sr)
        else:
            play_audio(wav, sr)

if __name__ == "__main__":
    main()
PYEOF
    chmod +x "$ENGINE_PY"
}

do_setup() {
    log_info "=== Qwen3-TTS Setup ==="
    install_uv
    create_venv
    install_deps
    create_engine
    log_info "=== Setup Complete ==="
    echo ""
    log_info "Test with: $0 \"Hello world\""
    log_info "See help:  $0 --help"
}

run_tts() {
    if ! check_setup; then
        log_warn "Setup not complete. Running setup first..."
        do_setup
    fi
    "$VENV_DIR/bin/python" "$ENGINE_PY" "$@"
}

show_help() {
    cat << 'HELPEOF'
Qwen3-TTS Helper

SETUP (one-time):
  ./qwen-tts.sh setup

BASIC USAGE:
  ./qwen-tts.sh "Hello world"              # Generate and play
  ./qwen-tts.sh "Hello" -o output.wav      # Save to file
  ./qwen-tts.sh "Hello" -o out.wav --play  # Save AND play

FILE INPUT (txt, pdf, docx, md):
  ./qwen-tts.sh -f document.pdf            # Read and play PDF
  ./qwen-tts.sh -f book.txt -o audio.wav   # Convert to audio file
  ./qwen-tts.sh -f readme.md --stream      # Stream markdown file

STREAMING (plays chunks as generated):
  ./qwen-tts.sh --stream "Long text. Multiple sentences."
  ./qwen-tts.sh -f novel.pdf --stream      # Stream long documents

VOICE SELECTION:
  ./qwen-tts.sh "Hello" -s serena          # Female voice
  ./qwen-tts.sh "Hello" -s uncle_fu        # Chinese male voice
  ./qwen-tts.sh --list-speakers            # Show all voices

AUDIO TUNING:
  --temperature 1.2       More varied/expressive (default: 1.0)
  --temperature 0.8       More consistent/monotone
  --top-p 0.95            Sampling diversity (default: 0.9)
  --repetition-penalty 1.1  Reduce repetition

LANGUAGES:
  ./qwen-tts.sh "Hello" -l Chinese
  ./qwen-tts.sh "Bonjour" -l French
  ./qwen-tts.sh --list-languages           # Show all languages

MODEL SIZE:
  -m 0.6B                 Faster, smaller (default)
  -m 1.7B                 Better quality, slower

SPEAKERS:
  aiden       Male, young, energetic
  dylan       Male, casual
  eric        Male, professional
  ono_anna    Female, Japanese accent
  ryan        Male, neutral (default)
  serena      Female, warm
  sohee       Female, Korean accent
  uncle_fu    Male, Chinese accent, mature
  vivian      Female, clear

EXAMPLES:
  ./qwen-tts.sh "Welcome!" -s vivian -o welcome.wav
  ./qwen-tts.sh -f report.pdf -o report.wav -s eric
  ./qwen-tts.sh --stream -f story.txt -s serena
  ./qwen-tts.sh "Hello" --temperature 1.3 -s aiden
HELPEOF
}

# Main
case "${1:-}" in
    setup)
        do_setup
        ;;
    -h|--help|help)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        run_tts "$@"
        ;;
esac
