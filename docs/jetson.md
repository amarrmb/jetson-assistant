# Jetson Setup Guide

This guide covers setting up Jetson Assistant on NVIDIA Jetson devices.

## Supported Devices

- Jetson Orin Nano
- Jetson Orin NX
- Jetson AGX Orin
- Jetson Xavier NX
- Jetson AGX Xavier

## Prerequisites

### JetPack

Ensure you have JetPack installed (5.x or 6.x recommended):

```bash
# Check JetPack version
cat /etc/nv_tegra_release
```

### System Packages

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils
```

## Installation

### Quick Install

```bash
git clone https://github.com/amarrmb/jetson-assistant.git
cd jetson-assistant
./scripts/install_jetson.sh
```

### Manual Install

1. Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install PyTorch for Jetson:

```bash
# For JetPack 6.x
pip install torch torchvision torchaudio \
    --index-url https://developer.download.nvidia.com/compute/redist/jp/v60
```

3. Install Jetson Assistant:

```bash
pip install -e ".[qwen,whisper]"
```

## Memory Management

### Check Available Memory

```bash
free -h
```

### Add Swap (Recommended for Large Models)

If you have less than 8GB RAM:

```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

### Model Size Recommendations

| Device | RAM | Recommended TTS | Recommended STT |
|--------|-----|-----------------|-----------------|
| Orin Nano (4GB) | 4GB | Piper | Whisper tiny |
| Orin Nano (8GB) | 8GB | Qwen 0.6B | Whisper base |
| Orin NX | 8-16GB | Qwen 0.6B | Whisper small |
| AGX Orin | 32-64GB | Qwen 1.7B | Whisper large |

## Power Modes

Jetson devices support different power modes. Check current mode:

```bash
sudo nvpmodel -q
```

For best TTS/STT performance, use maximum performance mode:

```bash
# AGX Orin - MAXN mode
sudo nvpmodel -m 0

# Orin NX - 25W mode
sudo nvpmodel -m 2
```

## GPU Memory

### Check GPU Memory

```bash
tegrastats
```

### Optimize GPU Usage

For memory-constrained devices, use smaller models or CPU fallback:

```python
from jetson_assistant import Engine

engine = Engine()

# Use smaller model
engine.load_tts_backend("qwen", model_size="0.6B")

# Or use CPU-only Piper
engine.load_tts_backend("piper")
```

## Running as a Service

### Create systemd Service

```bash
sudo nano /etc/systemd/system/jetson-assistant.service
```

```ini
[Unit]
Description=Jetson Assistant Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/jetson-assistant
Environment=PATH=/home/YOUR_USERNAME/jetson-assistant/.venv/bin
ExecStart=/home/YOUR_USERNAME/jetson-assistant/.venv/bin/jetson-assistant serve --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
sudo systemctl enable jetson-assistant
sudo systemctl start jetson-assistant
sudo systemctl status jetson-assistant
```

## Troubleshooting

### CUDA Out of Memory

Try smaller models or enable swap:

```bash
# Check GPU memory
nvidia-smi  # Not available on Jetson, use tegrastats instead
tegrastats

# Use smaller model
jetson-assistant tts "Hello" --model 0.6B
```

### Audio Not Playing

Check ALSA configuration:

```bash
# List audio devices
aplay -l

# Test audio
speaker-test -t wav -c 2
```

### Slow First Run

Model download and caching takes time on first run. Models are cached in `~/.cache/jetson-assistant/`.

### Permission Denied (GPU)

Ensure your user has GPU access:

```bash
sudo usermod -a -G video $USER
# Logout and login again
```

## Performance Tips

1. **Pre-warm models**: Load models at startup, not per-request
2. **Use streaming**: Stream long text instead of generating all at once
3. **Batch requests**: If processing multiple texts, keep the model loaded
4. **Monitor temps**: Use `tegrastats` to watch for thermal throttling
5. **Use power modes**: Higher power modes = faster processing
