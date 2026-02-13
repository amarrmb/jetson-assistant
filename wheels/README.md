# Pre-built Wheels for Jetson Thor

This directory contains pre-built Python wheels for Jetson Thor (sm_110 / Blackwell), used by the jetson-assistant Docker build to avoid lengthy compilation from source.

## Flash Attention 2.8.3

**File**: `flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl` (127MB)

Building flash-attn from source on Jetson Thor takes ~1 hour (or ~17 minutes with ninja). This pre-built wheel is included in the Docker build context so users don't have to wait.

### Compatibility Requirements

| Requirement | Value |
|-------------|-------|
| Python | 3.12 |
| Architecture | aarch64 (ARM64) |
| CUDA | 13.0 |
| GPU | Jetson Thor (sm_110 / Blackwell) |
| PyTorch | 2.9.1 (from jetson-ai-lab) |

### Distribution

The `.whl` file is **not stored in git** (127MB is too large). To obtain it:

1. **GitHub Releases** (recommended): Download from the jetson-assistant release assets
2. **Build from source**: See "Building Your Own Wheel" below
3. **Copy from Jetson-VLA**: `cp ~/baskd/Jetson-VLA/wheels/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl ./`

Place the wheel in this directory before running `docker build`.

### Manual Installation (without Docker)

```bash
# 1. Install PyTorch from Jetson AI Lab (MUST match wheel's PyTorch version)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# 2. Install the pre-built wheel
pip install flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl

# 3. Verify installation
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Important Notes

1. **PyTorch version must match**: The wheel was built against PyTorch 2.9.1. Using a different PyTorch version will cause `ImportError: libc10_cuda.so not found`.

2. **All Jetson Thor devices are compatible**: The wheel works on any Thor device since they all have:
   - Same CPU architecture (aarch64)
   - Same GPU (Blackwell sm_110)
   - Same CUDA version (13.0)

3. **Wheel is NOT in git**: The 127MB file is gitignored. Distribute via GitHub Releases or direct transfer.

### Verification Test

After installation, verify flash-attn works:
```bash
python -c "
import flash_attn
print(f'flash-attn version: {flash_attn.__version__}')
from flash_attn import flash_attn_func
print('flash_attn_func imported successfully')
"
```

## Building Your Own Wheel

If you need to rebuild (e.g., different PyTorch version):

```bash
# Ensure ninja is installed for parallel compilation
sudo apt-get install -y ninja-build

# Build wheel (~17 minutes with ninja)
FLASH_ATTN_CUDA_ARCHS='110' \
MAX_JOBS=10 \
NVCC_THREADS=2 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
    pip wheel flash-attn==2.8.3 --no-deps --no-cache-dir --no-build-isolation --wheel-dir ./
```

### Build Settings Explained

| Setting | Value | Description |
|---------|-------|-------------|
| `FLASH_ATTN_CUDA_ARCHS` | `110` | Target Thor's sm_110 only |
| `MAX_JOBS` | `10` | Parallel compilation jobs |
| `NVCC_THREADS` | `2` | Threads per nvcc (lower = more parallel jobs) |
| `ninja-build` | required | Enables parallel .cu file compilation |

Without ninja: ~2+ hours (sequential)
With ninja: ~17 minutes (parallel)
