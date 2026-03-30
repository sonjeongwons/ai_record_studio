# ================================================================
# AI Voice Studio - RunPod Serverless Docker Image (Optimized)
# ================================================================
# Target GPU: RTX 4090 (24GB VRAM, CUDA 12.1, SM 8.9 Ada Lovelace)
# Components: RVC v2 (Applio), Demucs, noisereduce, RunPod SDK
# Architecture: Multi-stage build (builder → runtime)
#   - Stage 1 (builder): devel image, compiles fairseq/pyworld/CUDA kernels
#   - Stage 2 (runtime): runtime image, copies compiled packages + models
# Build: docker build -t ai-voice-studio:latest .
# ================================================================


# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAGE 1: BUILDER — Compile all native/CUDA Python packages    ║
# ╚══════════════════════════════════════════════════════════════════╝
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# -- Prevent interactive prompts during apt-get --
ENV DEBIAN_FRONTEND=noninteractive

# -- Python optimization (no .pyc, unbuffered) --
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -- Compile CUDA kernels only for RTX 4090 (Ada Lovelace SM 8.9) --
ENV TORCH_CUDA_ARCH_LIST="8.9"

# ── 1.1 System dependencies (build tools + audio libs) ───────────
# Single RUN to minimize layers. --no-install-recommends to reduce size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # Build tools (C extensions: fairseq, pyworld, etc.)
        build-essential \
        cmake \
        pkg-config \
        # Python 3.10
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        # Audio processing
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        libsox-dev \
        sox \
        # Networking & download
        curl \
        wget \
        git \
        git-lfs \
        # Archive handling
        unzip \
        # Shared libraries (torch, fairseq, etc.)
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --no-cache-dir --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ── 1.2 PyTorch (pinned, CUDA 12.1) ─────────────────────────────
# ~2.5 GB — separate layer for cache efficiency.
# MUST be installed before Applio's requirements to prevent CPU-only torch.
RUN python -m pip install --no-cache-dir \
        torch==2.1.0+cu121 \
        torchvision==0.16.0+cu121 \
        torchaudio==2.1.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# -- Compatibility shim: Applio uses torch.amp.GradScaler (PyTorch 2.3+ API) --
# In PyTorch 2.1.x, GradScaler only lives in torch.cuda.amp.
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

# ── 1.3 Clone Applio (RVC v2) ───────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/IAHispano/Applio.git /app/Applio \
    # -- Fix PyTorch 2.1 compatibility in Applio source --
    && echo "--- Patching torch.amp references for PyTorch 2.1 ---" \
    && find /app/Applio -name "*.py" -print0 | xargs -0 sed -i \
        's/torch\.amp\.GradScaler/torch.cuda.amp.GradScaler/g' \
    && find /app/Applio -name "*.py" -print0 | xargs -0 sed -i \
        's/torch\.amp\.autocast("cuda"/torch.cuda.amp.autocast(/g' \
    && find /app/Applio -name "*.py" -print0 | xargs -0 sed -i \
        's/torch\.amp\.autocast(device_type="cuda"/torch.cuda.amp.autocast(/g' \
    && echo "--- Verifying patches ---" \
    && ! grep -r "torch\.amp\.GradScaler" /app/Applio/rvc/ --include="*.py" \
    && echo "All torch.amp references patched successfully" \
    # Clean up .git to save space (not needed at runtime)
    && rm -rf /app/Applio/.git

# ── 1.4 Applio Python Dependencies ──────────────────────────────
WORKDIR /app/Applio

# Install Applio requirements (filter out torch lines to protect our CUDA build)
RUN if [ -f requirements.txt ]; then \
        grep -vi "^torch" requirements.txt > /tmp/applio_filtered.txt 2>/dev/null; \
        python -m pip install --no-cache-dir -r /tmp/applio_filtered.txt || true; \
        rm -f /tmp/applio_filtered.txt; \
    fi \
    && if [ -d requirements ]; then \
        for req_file in requirements/*.txt; do \
            if [ -f "$req_file" ]; then \
                grep -vi "^torch" "$req_file" > /tmp/extra_req.txt 2>/dev/null; \
                python -m pip install --no-cache-dir -r /tmp/extra_req.txt || true; \
                rm -f /tmp/extra_req.txt; \
            fi; \
        done; \
    fi

# ── 1.5 Critical RVC runtime dependencies (pinned versions) ─────
# Group 1: numpy first (fairseq needs numpy<1.24)
RUN python -m pip install --no-cache-dir "numpy>=1.23.0,<1.24"

# Group 2: fairseq (notorious build issues — use fallback if official fails)
RUN python -m pip install --no-cache-dir fairseq==0.12.2 \
    || python -m pip install --no-cache-dir "git+https://github.com/facebookresearch/fairseq.git@v0.12.2" \
    || echo "WARNING: fairseq install failed, will rely on Applio bundled version"

# Group 3: faiss (cpu version avoids CUDA build issues)
RUN python -m pip install --no-cache-dir faiss-cpu==1.8.0

# Group 4: Audio/ML tools (pinned for reproducibility)
RUN python -m pip install --no-cache-dir \
        praat-parselmouth==0.4.4 \
        pyworld==0.3.4 \
        torchcrepe==0.0.22 \
        torchfcpe==0.0.4 \
        scipy==1.11.4 \
        librosa==0.10.2.post1 \
        soundfile==0.12.1 \
        pydub==0.25.1 \
        numba==0.59.1 \
        pedalboard==0.9.10

# Group 5: ONNX Runtime (pin version compatible with CUDA 12.1)
RUN python -m pip install --no-cache-dir onnxruntime-gpu==1.17.1 \
    || python -m pip install --no-cache-dir onnxruntime-gpu \
    || python -m pip install --no-cache-dir onnxruntime

# ── 1.6 Audio ML Tools (Demucs + noisereduce) ───────────────────
# Install demucs from GitHub main branch (v4.1.0a2+) — PyPI v4.0.1
# does NOT have demucs.api module which our handler requires.
# NOTE: pyannote.audio EXCLUDED — requires torchaudio>=2.2.0 which
#       conflicts with demucs (torchaudio<2.2) and torch 2.1.0.
#       Handler already handles pyannote absence gracefully (ImportError skip).
RUN python -m pip install --no-cache-dir \
        "git+https://github.com/adefossez/demucs.git" \
        noisereduce==3.0.2

# CRITICAL: Re-pin NumPy <2.0 AFTER all other installs.
# PyTorch 2.1.0 was compiled with NumPy 1.x C API.
RUN python -m pip install --no-cache-dir "numpy>=1.23.0,<2.0"

# ── 1.7 Applio implicit deps + RunPod SDK ───────────────────────
RUN python -m pip install --no-cache-dir \
        beautifulsoup4==4.12.3 \
        "transformers>=4.40.0,<4.45.0" \
        tensorboard==2.16.2 \
        wget==3.2 \
        runpod==1.8.1 \
        boto3==1.34.69 \
        requests==2.31.0

# -- Verify critical imports --
RUN python -c "from bs4 import BeautifulSoup; print('bs4 OK')" \
    && python -c "import transformers; print('transformers', transformers.__version__)" \
    && python -c "from transformers import HubertModel; print('HubertModel import OK')" \
    && python -c "import tensorboard; print('tensorboard OK')" \
    && python -c "import torch; print('torch', torch.__version__)" \
    && python -c "import numpy; print('numpy', numpy.__version__)"

# ── 1.8 Pre-download ALL ML Models (FlashBoot critical) ─────────
# Without pre-caching, first cold start downloads ~3GB of models
# and adds 30-120s latency. With pre-caching, cold start is <2s.

# -- RVC Pretrained v2 Models --
# v35: 40k를 KLM49_HFG (Korean Language Model)로 교체
# KLM49: 한국어 음소 최적화 (40p 음성학 스크립트 + 22명 성우/보컬리스트)
# 노래 데이터 포함 (남녀 저음~고음 전 음역), 한국어 노래에 최적
# by SeoulStreamingStation (Han TD) — 한국 RVC 커뮤니티 표준 pretrained
# 48k/32k도 KLM49 사용 (전 sample rate 지원)
RUN mkdir -p /app/Applio/rvc/models/pretraineds/pretrained_v2 \
    && cd /app/Applio/rvc/models/pretraineds/pretrained_v2 \
    && echo "Downloading KLM49_HFG Korean pretrained (40k, 48k, 32k)..." \
    && wget -q -O f0D40k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_40k.pth" \
    && wget -q -O f0G40k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_40k.pth" \
    && wget -q -O f0D48k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_48k.pth" \
    && wget -q -O f0G48k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_48k.pth" \
    && wget -q -O f0D32k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_32k.pth" \
    && wget -q -O f0G32k.pth \
       "https://huggingface.co/SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_32k.pth" \
    && echo "Pretrained v2 (all KLM49_HFG):" && ls -lh *.pth

# -- HuBERT / ContentVec Embedder (~1.2 GB) --
RUN mkdir -p /app/Applio/rvc/models/embedders/contentvec \
    && wget -q -O /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin" \
    && echo "ContentVec:" \
    && ls -lh /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin \
    # Symlink for legacy Applio code paths
    && mkdir -p /app/Applio/rvc/models/hubert \
    && ln -sf /app/Applio/rvc/models/embedders/contentvec/pytorch_model.bin \
              /app/Applio/rvc/models/hubert/hubert_base.pt

# -- RMVPE Model (~170 MB) --
RUN mkdir -p /app/Applio/rvc/models/predictors \
    && wget -q -O /app/Applio/rvc/models/predictors/rmvpe.pt \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt" \
    && echo "RMVPE:" \
    && ls -lh /app/Applio/rvc/models/predictors/rmvpe.pt \
    # Symlink for legacy paths
    && mkdir -p /app/Applio/rvc/models/rmvpe \
    && ln -sf /app/Applio/rvc/models/predictors/rmvpe.pt \
              /app/Applio/rvc/models/rmvpe/rmvpe.pt

# -- Demucs Models (~1.8 GB total) --
# htdemucs_6s: 6-stem model (drums/bass/other/vocals/guitar/piano)
#   v18: ft→6s — 피아노/기타 별도 분리 → 보컬 스템 누화 제거 → 화음 괴성 방지
# htdemucs_ft: 보조 다운로드 (캐시 충돌 방지)
ENV TORCH_HOME=/app/torch_hub
RUN mkdir -p /app/torch_hub \
    && python -c "\
from demucs.pretrained import get_model; \
print('Downloading htdemucs_6s (6-stem)...'); \
model = get_model('htdemucs_6s'); \
print('htdemucs_6s cached successfully.'); \
print('Downloading htdemucs_ft (fallback)...'); \
model2 = get_model('htdemucs_ft'); \
print('htdemucs_ft cached successfully.'); \
"

# -- Verify all models --
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAGE 2: RUNTIME — Minimal image with pre-built packages      ║
# ╚══════════════════════════════════════════════════════════════════╝
# Use runtime variant (no nvcc/dev headers). Saves ~4 GB vs devel.
# All CUDA kernels were already compiled in stage 1.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# ── Environment variables ────────────────────────────────────────
# Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# CUDA / GPU optimization
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Torch hub / Demucs model cache location
ENV TORCH_HOME=/app/torch_hub

# Applio root for our handler to locate RVC internals
ENV APPLIO_ROOT=/app/Applio

# Python import path
ENV PYTHONPATH="/app:/app/Applio"

# RunPod: ERROR for production, DEBUG for development
ENV RUNPOD_DEBUG_LEVEL=ERROR

# ── 2.1 Runtime system dependencies (no build tools) ────────────
# Only packages needed at runtime — no build-essential, cmake, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # Python 3.10 (runtime only, no -dev)
        python3.10 \
        python3.10-venv \
        python3-pip \
        # Audio processing (runtime)
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        # Networking (for model downloads at runtime if needed)
        curl \
        wget \
        # Shared libraries required by torch/fairseq/etc.
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ── 2.2 Copy compiled Python packages from builder ──────────────
# Copy the entire Python site-packages and scripts from builder stage.
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── 2.3 Copy Applio, models, and torch hub from builder ─────────
COPY --from=builder /app/Applio /app/Applio
COPY --from=builder /app/torch_hub /app/torch_hub

# ── 2.4 Create non-root user for security ───────────────────────
# Handler runs as unprivileged user. Models and workspace are
# owned by this user so the handler can read/write them.
RUN groupadd -r studio && useradd -r -g studio -m -s /bin/bash studio \
    && mkdir -p /app/workspace/audio \
                /app/workspace/models \
                /app/workspace/output \
                /app/workspace/tmp \
                /tmp/voice_studio \
    && chown -R studio:studio /app /tmp/voice_studio

# ── 2.5 Copy application handler (changes most often → last) ────
WORKDIR /app
COPY --chown=studio:studio runpod_handler.py /app/runpod_handler.py

# ── 2.6 Switch to non-root user ─────────────────────────────────
USER studio

# ── 2.7 Health check ────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import runpod; import torch; print('healthy')" || exit 1

# ── Entrypoint ──────────────────────────────────────────────────
# RunPod serverless manages networking — no ports to expose.
CMD ["python", "/app/runpod_handler.py"]
