# ================================================================
# AI Voice Studio - RunPod Serverless Docker Image
# ================================================================
# Target GPU: RTX 4090 (24GB VRAM, CUDA 12.1, SM 8.9 Ada Lovelace)
# Components: RVC v2 (Applio), Demucs, noisereduce, RunPod SDK
# Estimated image size: ~12-13 GB (all models pre-cached)
# Build: docker build -t ai-voice-studio:latest .
# CI/CD: GitHub Actions → ghcr.io (auto-build on push)
# ================================================================

# ── Base Image ─────────────────────────────────────────────────────
# devel variant required: fairseq compiles CUDA kernels at pip install
# time, and torch extensions may use JIT nvcc compilation at runtime.
# CUDA 12.1 matches RTX 4090 driver requirements.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ── Environment Variables ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
# No .pyc files (save space); unbuffered stdout (RunPod log streaming)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Compile CUDA kernels only for RTX 4090 (Ada Lovelace SM 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.9"
# Reduce CUDA memory fragmentation during long training runs
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Torch hub / Demucs model cache location
ENV TORCH_HOME=/app/torch_hub
# Applio root for our handler to locate RVC internals
ENV APPLIO_ROOT=/app/Applio
# Python import path
ENV PYTHONPATH="/app:/app/Applio:${PYTHONPATH}"
# RunPod: ERROR for production, DEBUG for development
ENV RUNPOD_DEBUG_LEVEL=ERROR


# ================================================================
# LAYER 1: System Dependencies
# ================================================================
# Single RUN to minimize layers. Grouped by purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # -- Build tools (C extensions: fairseq, pyworld, etc.) --
    build-essential \
    cmake \
    pkg-config \
    # -- Python 3.10 --
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # -- Audio processing --
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libsox-dev \
    sox \
    # -- Networking & download --
    curl \
    wget \
    git \
    git-lfs \
    # -- Archive handling --
    unzip \
    p7zip-full \
    # -- Shared libraries (torch, fairseq, etc.) --
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel


# ================================================================
# LAYER 2: PyTorch (pinned, CUDA 12.1)
# ================================================================
# ~2.5 GB - separate layer for cache efficiency (rarely changes).
# MUST be installed before Applio's requirements to prevent Applio
# from pulling CPU-only torch.
RUN python -m pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# -- Compatibility shim: Applio uses torch.amp.GradScaler (PyTorch 2.3+ API) --
# In PyTorch 2.1.x, GradScaler only lives in torch.cuda.amp.
# Patch torch/amp/__init__.py so `torch.amp.GradScaler` resolves correctly.
RUN python -c "\
import torch, os; \
amp_init = os.path.join(os.path.dirname(torch.__file__), 'amp', '__init__.py'); \
f = open(amp_init, 'a'); \
f.write('\n# Compat shim: GradScaler moved to torch.amp in PyTorch 2.3\n'); \
f.write('try:\n    from torch.cuda.amp import GradScaler\nexcept ImportError:\n    pass\n'); \
f.close(); \
import importlib; importlib.reload(torch.amp); \
assert hasattr(torch.amp, 'GradScaler'), 'Patch verification failed'; \
print('torch.amp.GradScaler shim installed OK') \
"


# ================================================================
# LAYER 3: Clone Applio (RVC v2)
# ================================================================
# Shallow clone saves ~800MB of git history.
# Applio is the actively maintained RVC v2 fork (MIT license).
WORKDIR /app
RUN git clone --depth 1 https://github.com/IAHispano/Applio.git /app/Applio


# ================================================================
# LAYER 4: Applio Python Dependencies
# ================================================================
WORKDIR /app/Applio

# 4a: Install Applio's main requirements.txt (torch lines filtered out
#     to protect our CUDA-compiled PyTorch from being overwritten)
RUN if [ -f requirements.txt ]; then \
        grep -vi "^torch" requirements.txt > /tmp/applio_filtered.txt 2>/dev/null; \
        python -m pip install --no-cache-dir -r /tmp/applio_filtered.txt || true; \
        rm -f /tmp/applio_filtered.txt; \
    fi

# 4b: Install any additional requirements files in requirements/ dir
RUN if [ -d requirements ]; then \
        for req_file in requirements/*.txt; do \
            if [ -f "$req_file" ]; then \
                grep -vi "^torch" "$req_file" > /tmp/extra_req.txt 2>/dev/null; \
                python -m pip install --no-cache-dir -r /tmp/extra_req.txt || true; \
                rm -f /tmp/extra_req.txt; \
            fi; \
        done; \
    fi

# 4c: Ensure all critical RVC runtime dependencies are present.
#     Split into groups to isolate build failures.

# -- Group 1: numpy first (fairseq needs numpy<1.24) --
RUN python -m pip install --no-cache-dir "numpy>=1.23.0,<1.24"

# -- Group 2: fairseq (notorious build issues — use omni-us fork if official fails) --
RUN python -m pip install --no-cache-dir fairseq==0.12.2 \
    || python -m pip install --no-cache-dir "git+https://github.com/facebookresearch/fairseq.git@v0.12.2" \
    || echo "WARNING: fairseq install failed, will rely on Applio bundled version"

# -- Group 3: faiss (cpu version is compatible and avoids CUDA build issues) --
RUN python -m pip install --no-cache-dir faiss-cpu

# -- Group 4: Audio/ML tools (pure Python or have wheels) --
RUN python -m pip install --no-cache-dir \
    praat-parselmouth \
    pyworld \
    torchcrepe \
    torchfcpe \
    scipy \
    librosa \
    soundfile \
    pydub \
    numba

# -- Group 5: ONNX Runtime (pin version compatible with CUDA 12.1) --
RUN python -m pip install --no-cache-dir onnxruntime-gpu==1.17.1 \
    || python -m pip install --no-cache-dir onnxruntime-gpu \
    || python -m pip install --no-cache-dir onnxruntime

# ================================================================
# LAYER 5: Audio ML Tools (Demucs + noisereduce)
# ================================================================
# Install demucs from GitHub main branch (v4.1.0a2+) — PyPI v4.0.1
# does NOT have demucs.api module which our handler requires.
RUN python -m pip install --no-cache-dir \
    "git+https://github.com/adefossez/demucs.git" \
    noisereduce

# -- CRITICAL: Re-pin NumPy <2.0 AFTER all other installs --
# PyTorch 2.1.0 was compiled with NumPy 1.x C API.
# demucs/librosa may pull NumPy 2.x which crashes torch at import.
RUN python -m pip install --no-cache-dir "numpy>=1.23.0,<2.0"


# ================================================================
# LAYER 6: Applio implicit deps + RunPod SDK
# ================================================================
# MUST be installed AFTER demucs/noisereduce to survive pip dependency resolution.
# beautifulsoup4: core.py → rvc/lib/tools/model_download.py → from bs4 import BeautifulSoup
# transformers: rvc/lib/utils.py → from transformers import HubertModel (ContentVec embedder)
# tensorboard: core.py → launch_tensorboard.py → from tensorboard import program
# wget: rvc/lib/utils.py → import wget
RUN python -m pip install --no-cache-dir \
    beautifulsoup4 \
    "transformers<4.45.0" \
    tensorboard \
    wget

RUN python -m pip install --no-cache-dir runpod

# -- Verify critical imports --
RUN python -c "from bs4 import BeautifulSoup; print('bs4 OK')" \
    && python -c "import transformers; print('transformers', transformers.__version__)" \
    && python -c "from transformers import HubertModel; print('HubertModel import OK')" \
    && python -c "import tensorboard; print('tensorboard OK')" \
    && python -c "import torch; print('torch', torch.__version__)" \
    && python -c "import numpy; print('numpy', numpy.__version__)"


# ================================================================
# LAYER 7: Pre-download ALL ML Models (FlashBoot critical)
# ================================================================
# Without pre-caching, first cold start downloads ~3GB of models
# and adds 30-120s latency. With pre-caching, cold start is <2s.

# -- 7a: RVC Pretrained v2 Models (~1.2 GB total) --
# Base generator (G) and discriminator (D) checkpoints for each
# sample rate. RVC fine-tunes from these during training.
# 40kHz = default for singing voice. 48kHz = highest quality.
RUN mkdir -p /app/Applio/rvc/models/pretraineds/pretrained_v2

RUN cd /app/Applio/rvc/models/pretraineds/pretrained_v2 \
    && wget -q -O f0D40k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D40k.pth" \
    && wget -q -O f0G40k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G40k.pth" \
    && wget -q -O f0D48k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D48k.pth" \
    && wget -q -O f0G48k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G48k.pth" \
    && wget -q -O f0D32k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D32k.pth" \
    && wget -q -O f0G32k.pth \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G32k.pth" \
    && echo "Pretrained v2:" && ls -lh *.pth

# -- 7b: HuBERT / ContentVec Embedder (~1.2 GB) --
# Extracts phonetic/speaker embeddings for RVC training + inference.
RUN mkdir -p /app/Applio/rvc/models/embedders/contentvec \
    && wget -q -O /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin" \
    && echo "ContentVec:" \
    && ls -lh /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin

# Symlink for legacy Applio code paths that check alternative locations
RUN mkdir -p /app/Applio/rvc/models/hubert \
    && ln -sf /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin \
              /app/Applio/rvc/models/hubert/hubert_base.pt

# -- 7c: RMVPE Model (~170 MB) --
# Robust Model for Vocal Pitch Estimation. Most accurate F0 extractor
# for singing voice -- superior to CREPE, PM, Harvest, DIO.
RUN mkdir -p /app/Applio/rvc/models/predictors \
    && wget -q -O /app/Applio/rvc/models/predictors/rmvpe.pt \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt" \
    && echo "RMVPE:" \
    && ls -lh /app/Applio/rvc/models/predictors/rmvpe.pt

# Symlink for legacy paths
RUN mkdir -p /app/Applio/rvc/models/rmvpe \
    && ln -sf /app/Applio/rvc/models/predictors/rmvpe.pt \
              /app/Applio/rvc/models/rmvpe/rmvpe.pt

# -- 7d: Demucs htdemucs_ft Model (~800 MB) --
# Hybrid Transformer Demucs (fine-tuned). Best vocal separation quality.
# Must be loaded via Python to populate torch hub cache correctly.
RUN mkdir -p /app/torch_hub \
    && python -c "\
from demucs.pretrained import get_model; \
print('Downloading htdemucs_ft...'); \
model = get_model('htdemucs_ft'); \
print('htdemucs_ft cached successfully.'); \
"

# -- 7e: Verify all models --
RUN echo "=== MODEL VERIFICATION ===" \
    && echo "-- pretrained_v2 --" \
    && ls -lh /app/Applio/rvc/models/pretraineds/pretrained_v2/*.pth \
    && echo "-- contentvec --" \
    && ls -lh /app/Applio/rvc/models/embedders/contentvec/ \
    && echo "-- rmvpe --" \
    && ls -lh /app/Applio/rvc/models/predictors/rmvpe.pt \
    && echo "-- demucs --" \
    && find /app/torch_hub -type f \( -name "*.th" -o -name "*.pt" -o -name "*.yaml" \) | head -10 \
    && echo "=== ALL MODELS VERIFIED ==="


# ================================================================
# LAYER 8: Application Code (LAST -- changes most frequently)
# ================================================================
# COPY is the final layer so handler edits only rebuild this layer.
# All expensive model downloads + pip installs above stay cached.
WORKDIR /app

RUN mkdir -p /app/workspace/audio \
             /app/workspace/models \
             /app/workspace/output \
             /app/workspace/tmp

COPY runpod_handler.py /app/runpod_handler.py


# ================================================================
# Runtime
# ================================================================
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import runpod; print('healthy')" || exit 1

# RunPod serverless manages networking -- no ports to expose.
CMD ["python", "/app/runpod_handler.py"]
