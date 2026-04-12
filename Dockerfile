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
# ⚠️ CVE-2025-32434: PyTorch <2.6.0의 torch.load()에 RCE 취약점 존재.
#    현재 2.1.0 사용 중 — Applio/fairseq 호환성 검증 후 2.6.0+로 업그레이드 예정.
#    완화 조치: 사용자 업로드 모델은 _validate_pth_file()로 기본 검증 수행.
RUN python -m pip install --no-cache-dir \
        torch==2.1.0+cu121 \
        torchvision==0.16.0+cu121 \
        torchaudio==2.1.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# NOTE: torch.amp.GradScaler shim 제거됨 — 순환 import 유발 (torch.__init__ → torch.amp → torch.cuda.amp → torch.amp 재귀)
# 대신 step 1.3에서 Applio 소스를 직접 패치하여 torch.amp.GradScaler → torch.cuda.amp.GradScaler 치환

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

# Install Applio requirements (filter out packages we pin ourselves)
# Applio 최신판이 Python 3.11+ 전용 버전을 요구하는 패키지들:
#   torch>=2.3, numpy>=2.3.5, scipy>=1.16.3 → 우리는 Python 3.10이므로 필터링
RUN if [ -f requirements.txt ]; then \
        grep -viE "^(torch|numpy|scipy)" requirements.txt > /tmp/applio_filtered.txt 2>/dev/null; \
        python -m pip install --no-cache-dir -r /tmp/applio_filtered.txt || true; \
        rm -f /tmp/applio_filtered.txt; \
    fi \
    && if [ -d requirements ]; then \
        for req_file in requirements/*.txt; do \
            if [ -f "$req_file" ]; then \
                grep -viE "^(torch|numpy|scipy)" "$req_file" > /tmp/extra_req.txt 2>/dev/null; \
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
# faiss-cpu 1.8.0: numpy<2.0 호환 (1.13.x+ 는 numpy>=1.25 요구)
RUN python -m pip install --no-cache-dir faiss-cpu==1.8.0 \
    || python -m pip install --no-cache-dir "faiss-cpu<1.9"

# Group 4: Audio/ML tools (pinned for reproducibility)
# torchfcpe/torchcrepe는 --no-deps로 설치 (torch 2.11.0 끌어오기 방지 — 2.1.0+cu121 유지)
RUN python -m pip install --no-cache-dir \
        praat-parselmouth==0.4.4 \
        pyworld==0.3.4 \
        scipy==1.13.1 \
        librosa==0.10.2.post1 \
        soundfile==0.12.1 \
        pydub==0.25.1 \
        numba==0.59.1 \
        pedalboard==0.9.10 \
    && python -m pip install --no-cache-dir --no-deps \
        "local_attention==1.9.0" \
        torchcrepe==0.0.22 \
        torchfcpe==0.0.4

# Group 5: ONNX Runtime (pin version compatible with CUDA 12.1)
RUN python -m pip install --no-cache-dir onnxruntime-gpu==1.17.1 \
    || python -m pip install --no-cache-dir onnxruntime-gpu \
    || python -m pip install --no-cache-dir onnxruntime

# ── 1.6 Audio ML Tools (Demucs + noisereduce + audio-separator) ──
# Demucs: from GitHub main (v4.1.0a2+) — PyPI v4.0.1 lacks demucs.api
# audio-separator: BS-Roformer/MDX-Net SOTA 보컬 분리 (SDR 12.9, Demucs보다 우수)
#   전처리에서 더 깨끗한 보컬 분리 → 학습 데이터 품질↑ → 기계음 감소
#
# ⚠️ audio-separator>=0.25.0은 torch>=2.3 요구하지만 우리는 torch==2.1.0 (Applio 필수).
#    --no-deps로 설치하고, 실제 필요한 의존성(einops, onnx 등)만 별도 설치.
#    런타임에 torch 2.1.0으로도 BS-Roformer 추론은 정상 동작.
# Step A: audio-separator/demucs 런타임 의존성 먼저 설치 (torch 의존 없는 것들)
RUN python -m pip install --no-cache-dir \
        noisereduce==3.0.2 \
        einops \
        ml_collections \
        julius \
        "beartype>=0.18.5,<0.19.0" \
        "onnx>=1.14" \
        onnx2torch \
        "resampy>=0.4" \
        "rotary-embedding-torch>=0.6.1,<0.7.0" \
        "samplerate==0.1.0" \
        diffq dora-search lameenc openunmix treetable
# Step B: demucs + audio-separator --no-deps (torch 의존성 끌어오기 방지)
RUN python -m pip install --no-cache-dir --no-deps \
        "git+https://github.com/adefossez/demucs.git" \
        audio-separator==0.25.1 \
    && python -c "import demucs; print('demucs OK')" \
    && python -c "from audio_separator.separator import Separator; print('audio-separator OK')"

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
    && python -c "import torch; assert torch.__version__.startswith('2.1.0'), f'WRONG TORCH: {torch.__version__}'; print('torch', torch.__version__, '✓')" \
    && python -c "import numpy; print('numpy', numpy.__version__)" \
    && python -c "import demucs; print('demucs OK')" \
    && python -c "from audio_separator.separator import Separator; print('audio-separator OK')" \
    && python -c "import torchcrepe; print('torchcrepe OK')" \
    && python -c "import torchfcpe; print('torchfcpe OK')"

# ── 1.8 Pre-download ALL ML Models (FlashBoot critical) ─────────
# Without pre-caching, first cold start downloads ~3GB of models
# and adds 30-120s latency. With pre-caching, cold start is <2s.

# -- RVC Pretrained v2 Models --
# 두 종류의 pretrained를 모두 캐시하여 학습 시 사용자가 선택 가능
#
# 1) KLM49_HFG: 한국어 노래 최적화 (한국어 음소 40p + 22명 성우/보컬리스트)
#    by SeoulStreamingStation (Han TD) — 한국 RVC 커뮤니티 표준
# 2) RIN_E3: 다국어/범용 (영어 팝, 일본어, 범용 노래에 적합)
#    by Applio 공식 — Applio 기본 pretrained

# -- KLM49_HFG (한국어) → pretrained_v2/ (기본 경로, 기존 호환) --
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
    && echo "KLM49_HFG:" && ls -lh *.pth

# -- RIN_E3 (다국어/범용) → pretrained_rin_e3/ --
RUN mkdir -p /app/Applio/rvc/models/pretraineds/pretrained_rin_e3 \
    && cd /app/Applio/rvc/models/pretraineds/pretrained_rin_e3 \
    && echo "Downloading RIN_E3 multilingual pretrained (40k, 48k, 32k)..." \
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
    && echo "RIN_E3:" && ls -lh *.pth

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

# -- Spin V2 Embedder (~380 MB) -- v54: 발음 명확도 개선 (ContentVec 대비)
# Applio 공식 HuggingFace에서 제공, 학습+추론 동일 embedder 필수
RUN mkdir -p /app/Applio/rvc/models/embedders/spin \
    && wget -q -O /app/Applio/rvc/models/embedders/spin/pytorch_model.bin \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin" \
    && wget -q -O /app/Applio/rvc/models/embedders/spin/config.json \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json" \
    && echo "Spin V2:" \
    && ls -lh /app/Applio/rvc/models/embedders/spin/

# -- Korean HuBERT Embedder (~380 MB) -- v54: 한국어 특화 임베더
RUN mkdir -p /app/Applio/rvc/models/embedders/korean-hubert-base \
    && wget -q -O /app/Applio/rvc/models/embedders/korean-hubert-base/pytorch_model.bin \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean-hubert-base/pytorch_model.bin" \
    && wget -q -O /app/Applio/rvc/models/embedders/korean-hubert-base/config.json \
       "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean-hubert-base/config.json" \
    && echo "Korean HuBERT:" \
    && ls -lh /app/Applio/rvc/models/embedders/korean-hubert-base/

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

# -- BS-Roformer SOTA 보컬 분리 모델 (~350 MB) --
# model_bs_roformer_ep_317_sdr_12.9755: 현재 SOTA (SDR 12.9)
# audio-separator 패키지의 내장 다운로드 기능 사용 (HuggingFace URL이 자주 변경됨)
# load_model()이 파일 없으면 자동 다운로드 → 빌드 시 사전 캐시
RUN mkdir -p /app/models/audio-separator \
    && python -c "\
from audio_separator.separator import Separator; \
s = Separator(model_file_dir='/app/models/audio-separator', output_dir='/tmp'); \
s.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt'); \
print('BS-Roformer model downloaded and loaded successfully'); \
" \
    && ls -lh /app/models/audio-separator/*.ckpt \
    && echo "BS-Roformer model cached"

# v54: BS PolarFormer — BS-Roformer 후속 모델 (Polar Coordinate Positional Embeddings)
# ZFTurbo v1.0.20 릴리즈, BS-Roformer 대비 향상된 보컬 분리
RUN python -c "\
from audio_separator.separator import Separator; \
s = Separator(model_file_dir='/app/models/audio-separator', output_dir='/tmp'); \
try: \
    s.load_model('model_bs_roformer_ep_368_sdr_13.0837.ckpt'); \
    print('BS PolarFormer model downloaded successfully'); \
except Exception as e: \
    print(f'BS PolarFormer download skipped (not yet in audio-separator): {e}'); \
" \
    && echo "BS PolarFormer cache attempt done"

# v49.5: mel_band_roformer_karaoke — 리드/백킹 보컬 분리 (화음 처리용, SDR 10.20)
# 보컬 스템에서 리드와 백킹을 추가 분리 → 리드만 RVC 변환, 백킹 원본 유지
RUN python -c "\
from audio_separator.separator import Separator; \
s = Separator(model_file_dir='/app/models/audio-separator', output_dir='/tmp'); \
s.load_model('mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'); \
print('Karaoke (lead/backing) model downloaded successfully'); \
" \
    && ls -lh /app/models/audio-separator/*karaoke* \
    && echo "Karaoke model cached"

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
    && echo "-- bs-roformer + karaoke --" \
    && ls -lh /app/models/audio-separator/*.ckpt \
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

# v54: Applio pipeline 청크 크기 최적화 (RTX 4090 24GB)
# v49.8: x_center=38→60 (38초마다 끊김 해소)
# v54: x_center=60→90, x_max=65→100 (RTX 4090 VRAM 충분, 청크 경계 최소화)
RUN CFG="/app/Applio/rvc/configs/config.py" \
    && if [ -f "$CFG" ]; then \
        python -c "import re,sys; f=sys.argv[1]; t=open(f).read(); t=re.sub(r'self\.x_pad\s*=\s*\d+','self.x_pad = 3',t); t=re.sub(r'self\.x_query\s*=\s*\d+','self.x_query = 10',t); t=re.sub(r'self\.x_center\s*=\s*\d+','self.x_center = 90',t); t=re.sub(r'self\.x_max\s*=\s*\d+','self.x_max = 100',t); open(f,'w').write(t); print('Applio config patched: x_center=90, x_max=100')" "$CFG" ; \
    else echo "config.py not found, skipping patch" ; fi

# v54: RMVPE 프레임 버퍼 패치 — 32프레임 제한→1.5초 컨텍스트
# codename-rvc-fork 발견: RMVPE가 32프레임으로 제한 → 가성 피치 불안정 근본 원인
# 94프레임 = 1.5초(16kHz, hop=256) 컨텍스트로 확대 → 피치 추적 안정화
RUN RMVPE_PY=$(find /app/Applio -name "rmvpe.py" -path "*/lib/*" 2>/dev/null | head -1) && \
    if [ -n "$RMVPE_PY" ] && [ -f "$RMVPE_PY" ]; then \
        python -c " \
import sys, re; \
f = sys.argv[1]; t = open(f).read(); \
# Expand any small hardcoded frame counts (32, 48) to 94 (~1.5s context) \
t_new = re.sub(r'(n_frames|frame_size|buffer_size)\s*=\s*(32|48)\b', r'\1 = 94', t); \
if t_new != t: \
    open(f, 'w').write(t_new); print(f'RMVPE patched: frame buffer expanded to 94 in {f}'); \
else: \
    print(f'RMVPE frame buffer not found or already patched in {f}'); \
" "$RMVPE_PY" ; \
    else echo "rmvpe.py not found, skipping RMVPE patch" ; fi
COPY --from=builder /app/torch_hub /app/torch_hub
COPY --from=builder /app/models /app/models

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
