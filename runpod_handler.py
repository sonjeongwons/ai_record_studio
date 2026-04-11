"""
AI Voice Studio - RunPod Serverless Handler
============================================
RunPod Serverless GPU (RTX 4090) 에서 실행되는 핸들러.
Docker 컨테이너 내부 /app/Applio/ 에 Applio 리포가 클론되어 있다고 가정.

지원하는 task_type:
  1. preprocess  - 오디오/영상 전처리 (추출, 분리, 다이어리제이션, 노이즈 제거, 세그먼트)
  2. train       - RVC v2 학습 (F0=RMVPE, Embedder=ContentVec, pretrained 사용)
  3. convert     - RVC v2 추론 (Singing Voice Conversion)

Docker 이미지에 필요한 것:
  - Applio repo at /app/Applio/
  - FFmpeg, Demucs (htdemucs_6s), noisereduce
  - pyannote-audio (optional, for speaker diarization — requires separate install)
  - CUDA 12.1+, PyTorch 2.x, FAISS-gpu
  - runpod SDK
  - pretrained models at /app/Applio/rvc/models/pretraineds/
"""

from __future__ import annotations

import os
import re
import sys
import gc
import json
import time
import uuid
import shutil
import base64
import logging
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import runpod

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPLIO_ROOT = Path("/app/Applio")
WORK_DIR = Path("/tmp/voice_studio")
PRETRAINED_DIR = APPLIO_ROOT / "rvc" / "models" / "pretraineds"

# Applio를 import path에 추가
sys.path.insert(0, str(APPLIO_ROOT))

# Segment duration bounds (seconds) — optimal for RVC training
SEGMENT_MIN = 3.0
SEGMENT_MAX = 12.0

# Audio extensions that are already audio (no video extraction needed)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("runpod_handler")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_dir(path: Path) -> None:
    """Recursively remove a directory, ignoring errors."""
    if path.exists():
        shutil.rmtree(str(path), ignore_errors=True)


def cleanup_gpu() -> None:
    """Release GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def decode_b64_file(data_b64: str, dest_path: Path) -> Path:
    """Decode a base64-encoded string and write it to dest_path."""
    try:
        raw = base64.b64decode(data_b64)
    except (base64.binascii.Error, ValueError) as e:
        raise ValueError(f"잘못된 base64 데이터입니다: {e}") from e
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(raw)
    return dest_path


def encode_file_b64(file_path: Path, compress: bool = False) -> str:
    """Read a file and return its base64 encoding, optionally gzip-compressed."""
    data = file_path.read_bytes()
    if compress:
        import gzip
        data = gzip.compress(data, compresslevel=6)
        log.info(
            f"Compressed {file_path.name}: "
            f"{file_path.stat().st_size / 1024 / 1024:.1f} MB → "
            f"{len(data) / 1024 / 1024:.1f} MB"
        )
    return base64.b64encode(data).decode("utf-8")


def _download_with_retries(
    requests_mod, url: str, dest: Path, label: str = "파일",
    timeout: int = 120, retries: int = 5
) -> None:
    """Download a file from URL with retry logic for transient network/HTTP errors.
    403/404 등 영구적 HTTP 오류는 재시도하지 않고 즉시 실패."""
    for attempt in range(retries):
        try:
            resp = requests_mod.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(resp.content)
            return
        except requests_mod.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
            # 403/404/401 = 영구적 오류 (만료 URL, 삭제된 파일 등) → 재시도 무의미, 즉시 실패
            if status_code in (401, 403, 404):
                raise RuntimeError(
                    f"{label} 다운로드 실패 (HTTP {status_code}): URL이 만료되었거나 파일이 존재하지 않습니다. "
                    f"모델을 다시 학습하거나 R2 설정을 확인하세요."
                ) from e
            # 그 외 HTTP 오류 (429, 500, 502 등) = 일시적 → 재시도
            if attempt < retries - 1:
                wait = min(2 ** attempt, 10)
                log.warning(f"{label} 다운로드 재시도 {attempt + 1}/{retries} (HTTP {status_code}, {wait}초 후): {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} 다운로드 {retries}회 실패 (HTTP {status_code}): {e}") from e
        except (requests_mod.exceptions.Timeout, requests_mod.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt, 10)
                log.warning(f"{label} 다운로드 재시도 {attempt + 1}/{retries} ({wait}초 후): {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} 다운로드 {retries}회 실패: {e}") from e
        except Exception as e:
            raise RuntimeError(f"{label} 다운로드 실패: {e}") from e


def get_audio_duration(file_path: str | Path) -> float:
    """Get duration of an audio file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        dur = float(result.stdout.strip())
        if dur <= 0:
            log.warning(f"ffprobe returned non-positive duration ({dur}) for {file_path}")
        return dur
    except Exception as e:
        log.warning(f"get_audio_duration failed for {file_path}: {e}")
        return 0.0


def run_ffmpeg(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with timeout."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    log.info(f"FFmpeg: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg 타임아웃 (>{timeout}초): {' '.join(args[:4])}...")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            log.error(f"FFmpeg stderr: {e.stderr[-500:]}")
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PREPROCESS task
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_preprocess(job_input: dict, job: dict) -> dict:
    """
    Full preprocessing pipeline:
      1) Decode incoming audio/video files
      2) Extract audio from video (FFmpeg)
      3) Vocal separation (Demucs htdemucs_6s)
      4) Speaker diarization (pyannote, optional)
      5) Noise reduction (noisereduce)
      6) Segment into 3-12s clips
      7) Return segments as base64 + metadata
    """
    audio_files: list[dict] = job_input.get("audio_files", [])
    if not audio_files:
        raise ValueError("No audio_files provided for preprocessing")

    job_id = job.get("id") or uuid.uuid4().hex[:12]
    work = ensure_dir(WORK_DIR / f"preprocess_{job_id}")
    raw_dir = ensure_dir(work / "raw")
    audio_dir = ensure_dir(work / "audio")
    vocal_dir = ensure_dir(work / "vocals")
    clean_dir = ensure_dir(work / "clean")
    segment_dir = ensure_dir(work / "segments")

    try:
        # --- Step 1: Decode files ---
        runpod.serverless.progress_update(job, "Decoding input files... (1/6)")
        raw_paths: list[Path] = []
        for i, fobj in enumerate(audio_files):
            fname = Path(fobj.get("filename", f"input_{i}.wav")).name  # path traversal 방지
            dest = raw_dir / fname
            decode_b64_file(fobj["data_base64"], dest)
            raw_paths.append(dest)
            log.info(f"Decoded file: {fname} ({dest.stat().st_size / 1024:.1f} KB)")

        # --- Step 2: Extract audio from video ---
        runpod.serverless.progress_update(job, "Extracting audio from video... (2/6)")
        audio_paths: list[Path] = []
        for rp in raw_paths:
            ext = rp.suffix.lower()
            if ext in VIDEO_EXTS:
                # Extract audio track to WAV 44.1kHz mono
                out_wav = audio_dir / (rp.stem + ".wav")
                run_ffmpeg([
                    "-i", str(rp),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "1",
                    str(out_wav),
                ])
                audio_paths.append(out_wav)
                log.info(f"Extracted audio from video: {rp.name} -> {out_wav.name}")
            elif ext in AUDIO_EXTS:
                # Convert to consistent WAV format
                out_wav = audio_dir / (rp.stem + ".wav")
                run_ffmpeg([
                    "-i", str(rp),
                    "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "1",
                    str(out_wav),
                ])
                audio_paths.append(out_wav)
            else:
                log.warning(f"Skipping unsupported file: {rp.name}")

        if not audio_paths:
            raise ValueError("No valid audio files after extraction")

        # --- Step 3: Vocal separation (BS-Roformer → Demucs fallback) ---
        # v36: BS-Roformer 우선 시도 (SDR 12.9, SOTA) → 더 깨끗한 학습 데이터
        vocal_paths = []
        accomp_paths = []
        _used_roformer = False
        for ap in audio_paths:
            try:
                runpod.serverless.progress_update(job, f"Separating vocals (BS-Roformer)... (3/6) - {ap.name}")
                roformer_out = ensure_dir(vocal_dir / "roformer")
                rf_result = _roformer_separate(ap, roformer_out)
                if rf_result.get("vocals") and rf_result["vocals"].exists():
                    vocal_paths.append(rf_result["vocals"])
                    if rf_result.get("accompaniment"):
                        accomp_paths.append(rf_result["accompaniment"])
                    _used_roformer = True
                    log.info(f"BS-Roformer separated: {ap.name}")
                    continue
            except Exception as rf_err:
                log.warning(f"BS-Roformer failed for {ap.name}: {rf_err}")
            # Demucs fallback
            log.info(f"Falling back to Demucs for {ap.name}")

        if not vocal_paths:
            runpod.serverless.progress_update(job, "Separating vocals (Demucs fallback)... (3/6)")
            separation = _demucs_separate(audio_paths, vocal_dir)
            vocal_paths = separation["vocals"]
            accomp_paths = separation["accompaniment"]
        elif _used_roformer:
            log.info("All files separated with BS-Roformer (SOTA quality)")

        # --- Step 4: Speaker diarization (optional) ---
        runpod.serverless.progress_update(job, "Speaker diarization... (4/6)")
        diarized_paths = _speaker_diarize(vocal_paths, work / "diarized")

        # --- Step 5: Noise reduction ---
        runpod.serverless.progress_update(job, "Noise reduction... (5/6)")
        cleaned_paths = _noise_reduce(diarized_paths, clean_dir)
        if not cleaned_paths:
            raise RuntimeError(
                f"노이즈 제거 후 유효한 오디오 파일이 없습니다. "
                f"입력 파일 {len(diarized_paths)}개 모두 손상되었거나 비정상적입니다."
            )

        # --- Step 5b: LUFS 정규화 (-23 LUFS, 커뮤니티 학습 데이터 표준) ---
        # AI Hub 권장: "HiFi-GAN needs perceptual quality → LUFS normalization"
        # -23 LUFS = 학습 데이터 표준 (-14 LUFS는 변환 출력용)
        runpod.serverless.progress_update(job, "Normalizing loudness (-23 LUFS)... (5.5/6)")
        normalized_paths = []
        for cp in cleaned_paths:
            norm_out = cp.with_suffix(".norm.wav")
            try:
                run_ffmpeg([
                    "-i", str(cp),
                    "-af", "loudnorm=I=-23:TP=-1:LRA=11",
                    "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    str(norm_out),
                ])
                if norm_out.exists() and norm_out.stat().st_size > 1000:
                    normalized_paths.append(norm_out)
                else:
                    normalized_paths.append(cp)
            except Exception as norm_err:
                log.warning(f"LUFS normalization failed for {cp.name}: {norm_err}")
                normalized_paths.append(cp)
        log.info(f"LUFS normalization: {len(normalized_paths)} files at -23 LUFS")
        cleaned_paths = normalized_paths

        # --- Step 5c: 학습 데이터 사전 디에싱 (v49.3) ---
        # AI Hub: "de-ess only actual sibilances, NOT blanket on entire dataset"
        # v49.1: blanket 6kHz -2dB → v49.3: 보수적 8kHz -1.0dB만 (금속성 아티팩트만 타겟)
        # 6kHz 대역은 자음 에너지 포함 → 건드리지 않음 (발음 보존)
        runpod.serverless.progress_update(job, "Light de-essing training data... (5.7/6)")
        deessed_paths = []
        for cp in cleaned_paths:
            de_out = cp.with_suffix(".de.wav")
            try:
                run_ffmpeg([
                    "-i", str(cp),
                    "-af",
                    "equalizer=f=8500:width_type=o:width=0.3:g=-1.0",
                    "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    str(de_out),
                ])
                if de_out.exists() and de_out.stat().st_size > 1000:
                    deessed_paths.append(de_out)
                else:
                    deessed_paths.append(cp)
            except Exception as de_err:
                log.warning(f"De-essing failed for {cp.name}: {de_err}")
                deessed_paths.append(cp)
        log.info(f"Training data de-essing v49.3: {len(deessed_paths)} files (8.5kHz -1.0dB only)")
        cleaned_paths = deessed_paths

        # --- Step 6: Segment into 3-12s clips ---
        runpod.serverless.progress_update(job, "Segmenting audio clips... (6/6)")
        segments, skipped_files = _segment_audio(cleaned_paths, segment_dir)
        if not segments:
            raise RuntimeError(
                "세그먼트 생성 실패: 유효한 세그먼트가 0개입니다. "
                "모든 오디오가 너무 짧거나 (< 2초) 손상되었습니다."
            )

        # Convert segments to FLAC (lossless) for transfer.
        # CRITICAL QUALITY DECISION: FLAC (lossless) vs MP3 (lossy)
        # MP3 192kbps introduces ~0.1-0.5% THD + frequency rolloff above 16kHz.
        # This gets baked into training data → model learns artifacts as voice features.
        # FLAC is lossless, ~50% smaller than WAV, and preserves ALL voice harmonics.
        # Result: better F0 extraction, more accurate ContentVec embeddings → natural output.
        segment_data = []
        total_duration = 0.0
        transfer_dir = ensure_dir(work / "flac_transfer")

        for seg_path in sorted(segment_dir.glob("*.wav")):
            dur = get_audio_duration(seg_path)
            total_duration += dur

            # Convert WAV → FLAC (lossless, ~50% size reduction vs WAV, no quality loss)
            flac_path = transfer_dir / (seg_path.stem + ".flac")
            try:
                run_ffmpeg([
                    "-i", str(seg_path),
                    "-acodec", "flac",
                    "-compression_level", "8",   # max FLAC compression (CPU intensive but smaller)
                    "-ar", "44100", "-ac", "1",
                    str(flac_path),
                ])
                encode_path = flac_path
            except Exception as e:
                log.warning(f"FLAC encode failed for {seg_path.name}: {e}, using WAV")
                encode_path = seg_path

            segment_data.append({
                "filename": encode_path.name,
                "path": str(encode_path),
                "duration_seconds": round(dur, 2),
            })

        # Export accompaniment (MR) files as FLAC for lossless mixing quality.
        # Accompaniment is used directly in _mix_audio() — lossless = better final mix.
        accomp_data = []
        for accomp_path in accomp_paths:
            dur = get_audio_duration(accomp_path)
            flac_path = transfer_dir / f"mr_{accomp_path.stem}.flac"
            try:
                run_ffmpeg([
                    "-i", str(accomp_path),
                    "-acodec", "flac",
                    "-compression_level", "8",
                    "-ar", "44100", "-ac", "2",
                    str(flac_path),
                ])
                encode_path = flac_path
            except Exception:
                encode_path = accomp_path
            accomp_data.append({
                "filename": encode_path.name,
                "path": str(encode_path),
                "duration_seconds": round(dur, 2),
            })

        # Export full vocal tracks as FLAC (lossless, for download and reference)
        vocal_data = []
        for vp in vocal_paths:
            dur = get_audio_duration(vp)
            flac_path = transfer_dir / f"vocal_{vp.stem}.flac"
            try:
                run_ffmpeg([
                    "-i", str(vp),
                    "-acodec", "flac",
                    "-compression_level", "8",
                    "-ar", "44100", "-ac", "1",
                    str(flac_path),
                ])
                encode_path = flac_path
            except Exception:
                encode_path = vp
            vocal_data.append({
                "filename": encode_path.name,
                "path": str(encode_path),
                "duration_seconds": round(dur, 2),
            })

        log.info(f"Preprocessed {len(segment_data)} segments ({total_duration:.1f}s), "
                 f"{len(accomp_data)} accompaniment, {len(vocal_data)} vocal files")

        # Upload all files to R2 bucket to avoid RunPod response payload limit (~20MB).
        # Inline base64 of segments+MR+vocals can easily exceed this for 2+ audio files.
        all_files = segment_data + accomp_data + vocal_data
        use_r2 = False
        try:
            from runpod.serverless.utils import upload_file_to_bucket
            bucket_name = job_input.get("bucket_name") or os.environ.get("BUCKET_NAME", "")
            if bucket_name and os.environ.get("BUCKET_ENDPOINT_URL"):
                use_r2 = True
        except ImportError:
            pass

        MAX_B64_BYTES = 15_000_000  # 15MB safety limit for RunPod payload
        total_b64_bytes = 0

        if use_r2:
            log.info(f"Uploading {len(all_files)} preprocessed files to R2...")
            for fobj in all_files:
                try:
                    url = upload_file_to_bucket(
                        file_name=fobj["filename"],
                        file_location=fobj["path"],
                        prefix=f"preprocess/{job_id}",
                        bucket_name=bucket_name,
                    )
                    if url and url.startswith("http"):
                        fobj["url"] = url
                    else:
                        log.warning(f"upload_file_to_bucket returned non-HTTP for "
                                    f"{fobj['filename']}: {str(url)[:120]}")
                        file_size = Path(fobj["path"]).stat().st_size
                        b64_size = (file_size * 4 + 2) // 3
                        if total_b64_bytes + b64_size <= MAX_B64_BYTES:
                            fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))
                            total_b64_bytes += b64_size
                        else:
                            log.warning(f"Skipping base64 for {fobj['filename']} "
                                        f"(would exceed {MAX_B64_BYTES // 1_000_000}MB limit)")
                except Exception as e:
                    log.warning(f"R2 upload failed for {fobj['filename']}: {e}")
                    file_size = Path(fobj["path"]).stat().st_size
                    b64_size = (file_size * 4 + 2) // 3
                    if total_b64_bytes + b64_size <= MAX_B64_BYTES:
                        fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))
                        total_b64_bytes += b64_size
                    else:
                        log.warning(f"Skipping base64 for {fobj['filename']} "
                                    f"(would exceed {MAX_B64_BYTES // 1_000_000}MB limit)")
            r2_count = sum(1 for f in all_files if "url" in f)
            b64_count = sum(1 for f in all_files if "data_base64" in f)
            skip_count = len(all_files) - r2_count - b64_count
            log.info(f"R2 upload complete: {r2_count} via URL, {b64_count} via base64 "
                     f"({total_b64_bytes / 1e6:.1f}MB), {skip_count} skipped")
        else:
            # No R2 configured — use inline base64 with size limit
            log.warning("No R2 bucket configured — using inline base64 for preprocess results")
            for fobj in all_files:
                file_size = Path(fobj["path"]).stat().st_size
                b64_size = (file_size * 4 + 2) // 3
                if total_b64_bytes + b64_size <= MAX_B64_BYTES:
                    fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))
                    total_b64_bytes += b64_size
                else:
                    log.warning(f"Skipping base64 for {fobj['filename']} "
                                f"(would exceed {MAX_B64_BYTES // 1_000_000}MB limit)")
            b64_count = sum(1 for f in all_files if "data_base64" in f)
            skip_count = len(all_files) - b64_count
            log.info(f"Base64 encoding: {b64_count} files ({total_b64_bytes / 1e6:.1f}MB), "
                     f"{skip_count} skipped due to size limit")

        # Remove local paths from response (not needed by server)
        for fobj in all_files:
            fobj.pop("path", None)

        result = {
            "segment_count": len(segment_data),
            "total_duration": round(total_duration, 2),
            "segments": segment_data,
            "accompaniment_files": accomp_data,
            "vocal_files": vocal_data,
            "skipped_files": skipped_files,
        }

        # Final payload size safety check
        result_json = json.dumps(result)
        result_size = len(result_json)
        log.info(f"Final result payload size: {result_size / 1e6:.2f} MB")

        if result_size > 18_000_000:
            log.error(f"Result payload too large ({result_size / 1e6:.1f} MB), "
                      f"stripping base64 data to prevent RunPod 400 error")
            for flist in [segment_data, accomp_data, vocal_data]:
                for fobj in flist:
                    if fobj.pop("data_base64", None):
                        fobj["data_stripped"] = True
            result_json = json.dumps(result)
            log.info(f"Trimmed payload size: {len(result_json) / 1e6:.2f} MB")

        return result

    finally:
        try:
            cleanup_dir(work)
        finally:
            cleanup_gpu()


def _roformer_separate(audio_path: Path, output_dir: Path) -> dict:
    """BS-Roformer SOTA 보컬 분리 (audio-separator 패키지, SDR 12.9).
    Demucs(SDR ~8.5)보다 훨씬 깨끗한 보컬 분리.
    Returns dict with "vocals" and "accompaniment" paths (or None).
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        log.warning("audio-separator not installed, falling back to Demucs")
        return {"vocals": None, "accompaniment": None}

    ensure_dir(output_dir)
    try:
        separator = Separator(
            output_dir=str(output_dir),
            output_format="wav",
            # BS-Roformer-ViperX-1297: 현재 SOTA 보컬 분리 모델
            model_file_dir="/app/models/audio-separator",
        )
        separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        result = separator.separate(str(audio_path))

        vocal_path = None
        accomp_path = None
        for f in result:
            fp = Path(f)
            if "vocal" in fp.name.lower() or "primary" in fp.name.lower():
                vocal_path = fp
            elif "instrument" in fp.name.lower() or "secondary" in fp.name.lower():
                accomp_path = fp

        if vocal_path and vocal_path.exists():
            # accompaniment 존재 여부도 검증
            if accomp_path and not accomp_path.exists():
                log.warning(f"BS-Roformer: accompaniment path found but file missing: {accomp_path}")
                accomp_path = None
            log.info(f"BS-Roformer separated: vocal={vocal_path.name}, "
                     f"accomp={accomp_path.name if accomp_path else 'N/A'}")
            return {"vocals": vocal_path, "accompaniment": accomp_path}
        else:
            log.warning("BS-Roformer produced no vocals output")
            return {"vocals": None, "accompaniment": None}
    except Exception as e:
        log.warning(f"BS-Roformer separation failed: {e}, falling back to Demucs")
        return {"vocals": None, "accompaniment": None}


def _separate_lead_backing(vocal_path: Path, output_dir: Path) -> dict:
    """v49.5: 리드 보컬 / 백킹 보컬 분리 (화음 처리용).

    BS-Roformer로 분리된 보컬 스템에서 리드와 백킹을 추가 분리.
    mel_band_roformer_karaoke 모델 사용 (리드 보컬 격리 특화).

    Returns dict:
      - "lead": 리드 보컬 경로 (RVC 변환 대상)
      - "backing": 백킹/화음 보컬 경로 (원본 유지 또는 경미 처리)
      - None 값이면 분리 실패 → 전체 보컬을 RVC 변환 (폴백)
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        log.warning("audio-separator not available for lead/backing separation")
        return {"lead": None, "backing": None}

    ensure_dir(output_dir)
    try:
        separator = Separator(
            output_dir=str(output_dir),
            output_format="wav",
            model_file_dir="/app/models/audio-separator",
        )
        # mel_band_roformer_karaoke: 리드 보컬 격리 SOTA (SDR 10.20)
        # 커뮤니티 검증: karaoke 모델의 primary=리드, secondary=백킹
        # dereverb는 다른 용도(리버브 제거)이므로 폴백에서 제외
        _LEAD_MODELS = [
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        ]
        _loaded = False
        for model_name in _LEAD_MODELS:
            try:
                separator.load_model(model_name)
                _loaded = True
                log.info(f"Lead/backing separation model: {model_name}")
                break
            except Exception as model_err:
                log.debug(f"Model {model_name} not available: {model_err}")

        if not _loaded:
            log.info("No lead/backing separation model available, skipping")
            return {"lead": None, "backing": None}

        result = separator.separate(str(vocal_path))

        lead_path = None
        backing_path = None
        for f in result:
            fp = Path(f)
            name_lower = fp.name.lower()
            # 모델마다 출력 파일명이 다르므로 여러 패턴 체크
            if any(k in name_lower for k in ["vocal", "primary", "lead"]):
                lead_path = fp
            elif any(k in name_lower for k in ["instrument", "secondary", "backing", "other"]):
                backing_path = fp

        if lead_path and lead_path.exists():
            log.info(f"Lead/backing separated: lead={lead_path.name}, "
                     f"backing={backing_path.name if backing_path else 'N/A'}")
            return {"lead": lead_path, "backing": backing_path}
        else:
            log.warning("Lead/backing separation produced no lead output")
            return {"lead": None, "backing": None}
    except Exception as e:
        log.warning(f"Lead/backing separation failed: {e}")
        return {"lead": None, "backing": None}


def _demucs_separate(audio_paths: list[Path], output_dir: Path) -> dict:
    """
    Run Demucs htdemucs_6s model to separate vocals from accompaniment (v18: 6-stem).
    Returns dict with:
      - "vocals": list of vocal-only WAV paths
      - "accompaniment": list of accompaniment (drums+bass+other) WAV paths

    Supports two backends:
      1. demucs.api (v4.1.0a2+ / GitHub main) — preferred, clean Python API
      2. demucs.pretrained + demucs.apply (v4.0.1 / PyPI) — fallback
    """
    import soundfile as sf
    import numpy as np

    ensure_dir(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_mono(arr):
        """Convert numpy array to mono 1D."""
        if arr.ndim == 2 and arr.shape[0] > 1:
            return arr.mean(axis=0)
        elif arr.ndim == 2:
            return arr[0]
        return arr

    def _to_stereo(arr):
        """Ensure numpy array is stereo [2, samples]. Preserves original stereo."""
        if arr.ndim == 1:
            return np.stack([arr, arr])  # mono → duplicate to stereo
        if arr.ndim == 2 and arr.shape[0] == 1:
            return np.concatenate([arr, arr], axis=0)
        return arr  # already stereo or multi-channel

    def _safe_sum_stems(stems_list):
        """Sum multiple stems and normalize to prevent clipping.
        Returns stereo numpy array with peak at -1dB headroom."""
        result = None
        for s in stems_list:
            s_stereo = _to_stereo(s)
            result = s_stereo if result is None else result + s_stereo
        if result is not None:
            peak = np.abs(result).max()
            if peak > 0:
                result = result * (10.0 ** (-1 / 20) / peak)  # 정확한 -1 dBFS = 0.89125
        return result

    # --- Try demucs.api first (GitHub v4.1.0a2+) ---
    try:
        import demucs.api
        log.info("Using demucs.api (v4.1+) for vocal separation (shifts=5, overlap=0.25)")
        # seed 파라미터는 demucs 일부 버전에서 미지원 — 대신 torch.manual_seed로 재현성 확보
        torch.manual_seed(0)
        separator = demucs.api.Separator(
            model="htdemucs_6s",    # v18: ft→6s — 피아노/기타 별도 스템 분리 → 보컬 누화 제거
            device=device,
            shifts=5,       # 5 random time shifts → average = dramatically better SDR
            overlap=0.25,   # v45: 0.6→0.25 (타이밍 드리프트/스미어링 감소, 더블링 방지)
        )

        vocal_paths: list[Path] = []
        accomp_paths: list[Path] = []
        for audio_path in audio_paths:
            try:
                origin, separated = separator.separate_audio_file(str(audio_path))
                if "vocals" in separated:
                    # Save vocals as MONO (RVC requires mono input)
                    vocal_wav = output_dir / f"{audio_path.stem}_vocals.wav"
                    vocal_np = _to_mono(separated["vocals"].cpu().numpy())
                    # FLOAT: Demucs float32 출력을 그대로 보존 (PCM_16 대비 양자화 손실 제거)
                    sf.write(str(vocal_wav), vocal_np, samplerate=44100, subtype="FLOAT")
                    vocal_paths.append(vocal_wav)

                    # Build accompaniment = drums + bass + other + guitar + piano (STEREO preserved)
                    # v18: htdemucs_6s — guitar/piano 스템도 MR에 포함
                    #      (보컬 스템에서 분리된 피아노/기타를 반주에 복원)
                    mr_stems = []
                    for stem in ("drums", "bass", "other", "guitar", "piano"):
                        if stem in separated:
                            mr_stems.append(separated[stem].cpu().numpy())
                    if mr_stems:
                        accomp_np = _safe_sum_stems(mr_stems)
                        accomp_wav = output_dir / f"{audio_path.stem}_accompaniment.wav"
                        # Transpose [channels, samples] → [samples, channels] for soundfile
                        # FLOAT: 반주 트랙도 float32 보존 (믹싱 시 최대 다이나믹레인지)
                        sf.write(str(accomp_wav), accomp_np.T, samplerate=44100, subtype="FLOAT")
                        accomp_paths.append(accomp_wav)

                    log.info(f"Demucs separation done: {audio_path.name} → vocals (mono) + accompaniment (stereo)")
                else:
                    log.warning(f"No 'vocals' source from Demucs for {audio_path.name}, using original")
                    vocal_paths.append(audio_path)
            except Exception as e:
                log.error(f"Demucs failed for {audio_path.name}: {e}")
                vocal_paths.append(audio_path)

        cleanup_gpu()
        return {"vocals": vocal_paths, "accompaniment": accomp_paths}

    except ImportError:
        log.warning("demucs.api not available (PyPI v4.0.1), using fallback with demucs.pretrained + demucs.apply")

    # --- Fallback: demucs.pretrained + demucs.apply (PyPI v4.0.1) ---
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    model = get_model("htdemucs_6s")    # v18: ft→6s — 피아노/기타 별도 분리
    model.to(device)
    # htdemucs_6s sources order: drums, bass, other, vocals, guitar, piano
    source_names = model.sources
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1

    vocal_paths: list[Path] = []
    accomp_paths: list[Path] = []
    for audio_path in audio_paths:
        try:
            wav = AudioFile(str(audio_path)).read(
                streams=0, samplerate=model.samplerate, channels=model.audio_channels
            )
            ref = wav.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            if ref_std < 1e-8:  # 무음 오디오 — 정규화 생략
                ref_std = torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
            wav = (wav - ref_mean) / ref_std
            sources = apply_model(model, wav[None], device=device, shifts=5, overlap=0.25)[0]
            sources = sources * ref_std + ref_mean

            if vocals_idx >= 0:
                # Save vocals as MONO (RVC requires mono input)
                vocal_wav_path = output_dir / f"{audio_path.stem}_vocals.wav"
                vocal_np = _to_mono(sources[vocals_idx].cpu().numpy())
                sf.write(str(vocal_wav_path), vocal_np, samplerate=model.samplerate, subtype="FLOAT")
                vocal_paths.append(vocal_wav_path)

                # Build accompaniment: sum all non-vocal sources (STEREO preserved)
                mr_stems = []
                for i, sname in enumerate(source_names):
                    if sname != "vocals":
                        mr_stems.append(sources[i].cpu().numpy())
                if mr_stems:
                    accomp_np = _safe_sum_stems(mr_stems)
                    accomp_wav_path = output_dir / f"{audio_path.stem}_accompaniment.wav"
                    sf.write(str(accomp_wav_path), accomp_np.T, samplerate=model.samplerate, subtype="FLOAT")
                    accomp_paths.append(accomp_wav_path)

                log.info(f"Demucs separation done (fallback): {audio_path.name} → vocals (mono) + accompaniment (stereo)")
            else:
                log.warning(f"No 'vocals' source in model for {audio_path.name}, using original")
                vocal_paths.append(audio_path)
        except Exception as e:
            log.error(f"Demucs fallback failed for {audio_path.name}: {e}", exc_info=True)
            cleanup_gpu()  # OOM 등 발생 시 GPU 메모리 즉시 해제
            vocal_paths.append(audio_path)

    cleanup_gpu()
    return {"vocals": vocal_paths, "accompaniment": accomp_paths}


def _speaker_diarize(vocal_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    Speaker diarization using pyannote/speaker-diarization-3.1.
    If only 1 speaker detected, skip diarization and return original files.
    Extracts segments for the dominant speaker only.
    """
    ensure_dir(output_dir)

    # Check if HF token is available for pyannote (gated model)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN not set, skipping speaker diarization")
        return vocal_paths

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        import soundfile as sf

        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        result_paths: list[Path] = []
        for vp in vocal_paths:
            diarization = pipeline(str(vp))

            # Count speakers
            speakers = set()
            speaker_durations: dict[str, float] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                dur = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + dur

            if len(speakers) <= 1:
                log.info(f"Only {len(speakers)} speaker(s) in {vp.name}, no diarization needed")
                result_paths.append(vp)
                continue

            if not speaker_durations:
                log.warning("No speakers detected, returning input as-is")
                result_paths.append(vp)
                continue

            # Find dominant speaker (longest total duration)
            dominant = max(speaker_durations, key=speaker_durations.get)
            log.info(f"Found {len(speakers)} speakers in {vp.name}, extracting dominant: {dominant}")

            # Read full audio
            audio_data, sr = sf.read(str(vp))

            # Extract dominant speaker segments and concatenate
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == dominant:
                    start_sample = int(turn.start * sr)
                    end_sample = int(turn.end * sr)
                    if audio_data.ndim == 1:
                        segments.append(audio_data[start_sample:end_sample])
                    else:
                        segments.append(audio_data[start_sample:end_sample, :])

            if segments:
                concatenated = np.concatenate(segments, axis=0)
                out_path = output_dir / f"{vp.stem}_speaker.wav"
                sf.write(str(out_path), concatenated, samplerate=sr, subtype="FLOAT")
                result_paths.append(out_path)
            else:
                result_paths.append(vp)

        cleanup_gpu()
        return result_paths

    except ImportError:
        log.warning("pyannote.audio not installed, skipping diarization")
        return vocal_paths
    except Exception as e:
        log.error(f"Speaker diarization failed: {e}")
        return vocal_paths


def _noise_reduce(audio_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    Adaptive noise reduction using noisereduce library.
    SNR 기반으로 NR 강도를 자동 조절하여 최대 자연스러움 보존:

    - 깨끗한 신호 (>-20dB): NR 스킵 → 하모닉스/숨소리 완전 보존
    - 보통 신호 (-30~-20dB): prop_decrease=0.25 (약한 NR)
    - 노이즈 있음 (<-30dB): prop_decrease=0.35 (중간 NR)
    - stationary=True: Demucs 이후 잔존 노이즈는 험/히스 → 정상 노이즈만 타겟
    - n_fft=4096: 고해상도 FFT → 하모닉과 노이즈 정밀 분리
    - hop_length=256: 미세 주파수 분석 → 음성 하모닉 보존
    """
    ensure_dir(output_dir)

    try:
        import noisereduce as nr
        import soundfile as sf
        import numpy as np

        cleaned: list[Path] = []
        skipped_clean = 0
        for ap in audio_paths:
            audio_data, sr = sf.read(str(ap))
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                log.warning(f"Skipping corrupted audio file: {ap.name}")
                continue

            # SNR 기반 적응형 NR 강도 결정
            # 전체 RMS가 아닌 실제 SNR 추정: 가장 조용한 10% 프레임 = 노이즈 추정
            # → 조용하지만 깨끗한 가성/소프트 보컬을 잘못 NR하는 문제 방지
            _frame_size = max(1, int(sr * 0.02))  # 20ms 프레임
            _n_frames = max(1, len(audio_data) // _frame_size)
            _frame_rms = np.array([
                np.sqrt(np.mean(audio_data[i*_frame_size:(i+1)*_frame_size] ** 2))
                for i in range(_n_frames)
            ])
            _frame_rms_sorted = np.sort(_frame_rms)
            _noise_rms = float(np.mean(_frame_rms_sorted[:max(1, _n_frames // 10)]))  # 하위 10%
            _signal_rms = float(np.sqrt(np.mean(audio_data ** 2)))
            snr_db = 20 * np.log10(max(_signal_rms, 1e-10) / max(_noise_rms, 1e-10))

            out_path = output_dir / f"{ap.stem}_clean.wav"

            if snr_db > 20:
                # v36: 임계값 25→20dB — BS-Roformer/Demucs 분리 후에는 대부분 깨끗
                # 불필요한 NR을 줄여 숨소리/자음 질감 최대 보존
                sf.write(str(out_path), audio_data, samplerate=sr, subtype="FLOAT")
                cleaned.append(out_path)
                skipped_clean += 1
                log.info(f"Clean signal (SNR={snr_db:.1f}dB), NR skipped: {ap.name}")
                continue

            if snr_db > 15:
                prop = 0.06  # v45: 0.10→0.06 — 자음 에너지 보호 (P/T/K 파열음, S/Sh 치찰음)
            else:
                prop = 0.12  # v45: 0.18→0.12 — 약한 NR, 자음/숨소리 보호 강화

            # n_fft/hop_length를 샘플레이트에 정규화 (93ms/12ms 타겟)
            _nr_nfft = max(256, 2 ** (int(sr * 0.093).bit_length() - 1))  # ~93ms, 2의 거듭제곱
            _nr_hop = max(64, int(sr * 0.012))  # ~12ms
            reduced = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                prop_decrease=prop,
                stationary=True,    # v45: 정상 노이즈 가정 (자음을 노이즈로 오분류 방지)
                n_fft=_nr_nfft,     # SR 정규화: 44.1k→4096, 48k→4096, 16k→1024
                hop_length=_nr_hop, # SR 정규화: 44.1k→529, 48k→576, 16k→192
                freq_mask_smooth_hz=1000,
            )

            sf.write(str(out_path), reduced, samplerate=sr, subtype="FLOAT")
            cleaned.append(out_path)
            log.info(f"NR done (prop={prop}, SNR={snr_db:.1f}dB): {ap.name}")

        if skipped_clean:
            log.info(f"Adaptive NR: {skipped_clean}/{len(audio_paths)} files clean enough to skip")
        return cleaned

    except ImportError:
        log.warning("noisereduce not installed, skipping noise reduction")
        return audio_paths
    except Exception as e:
        log.error(f"Noise reduction failed: {e}")
        return audio_paths


def _segment_audio(audio_paths: list[Path], output_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Split audio files into 3-12 second segments using silence detection.
    Uses FFmpeg silencedetect to find natural split points,
    falling back to fixed-length splitting.
    Returns (segments, skipped_files) where skipped_files lists corrupted file names.
    """
    import soundfile as sf

    ensure_dir(output_dir)
    all_segments: list[Path] = []
    skipped_files: list[str] = []
    seg_idx = 0

    for ap in audio_paths:
        # 원본 파일명에서 처리 접미사들을 모두 제거하여 소스 stem 추출
        # 순서 중요: _diarized → _clean → _vocals (역순으로 붙었으므로)
        source_stem = ap.stem
        for suffix in ("_speaker", "_clean", "_vocals"):
            if source_stem.endswith(suffix):
                source_stem = source_stem[:-len(suffix)]

        if not ap.exists():
            log.error(f"오디오 파일이 없습니다 (건너뜀): {ap}")
            skipped_files.append(ap.name)
            continue
        audio_data, sr = sf.read(str(ap))
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            log.warning(f"Skipping corrupted audio file: {ap.name}")
            skipped_files.append(ap.name)
            continue
        total_samples = len(audio_data)
        total_duration = total_samples / sr

        if total_duration <= SEGMENT_MAX:
            # File is already short enough, keep as-is if >= SEGMENT_MIN
            if total_duration >= SEGMENT_MIN:
                out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
                sf.write(str(out), audio_data, samplerate=sr, subtype="FLOAT")
                all_segments.append(out)
                seg_idx += 1
            elif total_duration >= 2.0:
                # Keep very short clips too (will be less than 5s but still usable)
                out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
                sf.write(str(out), audio_data, samplerate=sr, subtype="FLOAT")
                all_segments.append(out)
                seg_idx += 1
            continue

        # Try silence-based splitting first
        split_points = _detect_silence_splits(str(ap), sr, total_duration)

        if not split_points:
            # Fallback: fixed 10-second segments
            split_points = []
            pos = 0.0
            while pos < total_duration:
                end = min(pos + 10.0, total_duration)
                remaining = end - pos
                if remaining >= 2.0:
                    split_points.append((pos, end))
                elif remaining > 0 and split_points:
                    # Merge short tail into preceding segment to avoid data loss
                    prev_start, _ = split_points[-1]
                    split_points[-1] = (prev_start, end)
                pos = end

        for start_sec, end_sec in split_points:
            start_s = int(start_sec * sr)
            end_s = min(int(end_sec * sr), total_samples)
            segment = audio_data[start_s:end_s].copy()

            # 30ms Hann 창 페이드인/아웃 → 세그먼트 경계 클릭 방지
            # 선형(linspace) → Hann half-window: 에너지 연속성 보장, 스펙트럼 누설 감소
            # 30ms는 1 글로탈 사이클(남성: 5-15ms)의 2-6배 → 자연스러운 에너지 전환
            fade_samples = min(int(sr * 0.03), len(segment) // 6)
            if fade_samples > 0:
                # Hann window half: 0→1 (fade-in) and 1→0 (fade-out)
                hann = np.hanning(fade_samples * 2)
                fade_in = hann[:fade_samples]
                fade_out = hann[fade_samples:]
                segment[:fade_samples] *= fade_in
                segment[-fade_samples:] *= fade_out

            out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
            sf.write(str(out), segment, samplerate=sr, subtype="FLOAT")
            all_segments.append(out)
            seg_idx += 1

    log.info(f"Total segments created: {len(all_segments)}, skipped: {len(skipped_files)}")

    # v46: 무음 세그먼트 자동 제거 (RMS < -40dBFS)
    # 분석 결과 7.2% 세그먼트가 무음 → 학습 용량 낭비 + 보코더 혼란
    filtered = []
    removed_silent = 0
    for seg in all_segments:
        try:
            _seg_data, _seg_sr = sf.read(str(seg))
            _seg_rms = float(np.sqrt(np.mean(_seg_data ** 2)))
            if _seg_rms < 0.01:  # -40 dBFS 이하
                seg.unlink(missing_ok=True)
                removed_silent += 1
                continue
        except Exception:
            pass
        filtered.append(seg)
    if removed_silent:
        log.info(f"Removed {removed_silent} near-silent segments (RMS < -40dBFS)")
    all_segments = filtered

    return all_segments, skipped_files


def _detect_silence_splits(
    audio_path: str, sr: int, total_duration: float
) -> list[tuple[float, float]]:
    """
    Use FFmpeg silencedetect to find silence boundaries,
    then create segments of 3-12 seconds at those boundaries.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", audio_path,
                "-af", "silencedetect=noise=-45dB:d=0.4",
                # -45dB: 부드러운 가성/여린 노래에서 오탐지 방지 (-35dB는 너무 공격적)
                # d=0.4: 짧은 강세 사이 숨소리를 무음으로 오해하지 않도록
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=120,
        )
        stderr = result.stderr

        # Parse silence_end timestamps
        silence_ends: list[float] = []
        for line in stderr.split("\n"):
            if "silence_end" in line:
                try:
                    # Format: [silencedetect @ ...] silence_end: 1.234 | silence_duration: 0.567
                    parts = line.split("silence_end:")[1].split("|")[0].strip()
                    silence_ends.append(float(parts))
                except (IndexError, ValueError):
                    continue

        if not silence_ends:
            return []

        # Build segments using silence points as potential split boundaries
        segments: list[tuple[float, float]] = []
        seg_start = 0.0

        for sil_end in silence_ends:
            seg_len = sil_end - seg_start
            if seg_len >= SEGMENT_MAX:
                # Force split at SEGMENT_MAX boundary
                while seg_start + SEGMENT_MAX < sil_end:
                    segments.append((seg_start, seg_start + SEGMENT_MAX))
                    seg_start += SEGMENT_MAX
                if sil_end - seg_start >= SEGMENT_MIN:
                    segments.append((seg_start, sil_end))
                    seg_start = sil_end
            elif seg_len >= SEGMENT_MIN:
                segments.append((seg_start, sil_end))
                seg_start = sil_end
            # else: too short, continue accumulating

        # Handle remaining audio
        remaining = total_duration - seg_start
        if remaining >= SEGMENT_MIN:
            segments.append((seg_start, total_duration))
        elif remaining >= 2.0 and segments:
            # Merge with last segment if it won't exceed max
            last_start, _ = segments[-1]
            if (total_duration - last_start) <= SEGMENT_MAX * 1.2:
                segments[-1] = (last_start, total_duration)
            else:
                segments.append((seg_start, total_duration))
        elif remaining >= 2.0:
            segments.append((seg_start, total_duration))

        return segments

    except Exception as e:
        log.warning(f"Silence detection failed: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TRAIN task
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_train(job_input: dict, job: dict) -> dict:
    """
    RVC v2 training pipeline using Applio's internal modules:
      1) Decode audio files
      2) Preprocess (resample to target SR)
      3) Extract F0 (RMVPE)
      4) Extract features (ContentVec / HuBERT)
      5) Train model (configurable epochs, batch_size)
      6) Generate FAISS index
      7) Return .pth + .index as base64
    """
    model_name: str = job_input.get("model_name", "my_voice_model")
    audio_files: list[dict] = job_input.get("audio_files", [])
    audio_urls: list[dict] = job_input.get("audio_urls", [])
    try:
        sample_rate: int = int(job_input.get("sample_rate", 40000))
    except (ValueError, TypeError):
        sample_rate = 40000  # v32: 48k→40k 복원 (48k 보코더 기계음 이슈 — RVC Issue #119/#514)
    # v36: 200에폭 기본 (150에서 상향) — 체크포인트 비교로 최적점 탐색
    # 과학습 감지기(50 threshold)가 자동 보호. save_every=10으로 매 10에폭 저장
    try:
        epochs: int = int(job_input.get("epochs", 150))  # v49: 200→150 (과적합 방지)
    except (ValueError, TypeError):
        epochs = 150
    # v36: batch 8 기본 (4에서 상향) — 43분+ 데이터에 더 안정적 (커뮤니티 권장)
    # RTX 4090 24GB VRAM에서 batch 8 안정 동작
    try:
        batch_size: int = int(job_input.get("batch_size", 8))  # v49.4: →8 (AI Hub 명확: >30분=8)
    except (ValueError, TypeError):
        batch_size = 8  # v49.7: 폴백도 8 (AI Hub: >30분=8)
    _VALID_F0 = {"rmvpe", "fcpe", "crepe", "crepe-tiny", "harvest", "pm"}
    f0_method: str = job_input.get("f0_method", "rmvpe")
    if f0_method not in _VALID_F0:
        log.warning(f"Invalid f0_method '{f0_method}', falling back to rmvpe")
        f0_method = "rmvpe"
    embedder_model: str = job_input.get("embedder_model", "contentvec")
    # pretrained 모델 선택: "klm49" (한국어) 또는 "rin_e3" (다국어/팝송)
    pretrained_model: str = job_input.get("pretrained_model", "klm49")
    # v35: 25→10 — 200 epoch에서 20개 체크포인트로 최적 epoch 정밀 식별
    save_every_epoch: int = max(1, int(job_input.get("save_every_epoch", 10)))

    if not audio_files and not audio_urls:
        raise ValueError("No audio_files or audio_urls provided for training")

    # Validate model name — must be safe for filesystem paths
    model_name = re.sub(r'[<>:"/\\|?*]', '_', model_name).strip()
    if not model_name:
        model_name = "my_voice_model"
    if len(model_name) > 64:
        model_name = model_name[:64]

    # Validate parameters — Applio only supports 32k, 40k, 48k
    if sample_rate not in (32000, 40000, 48000):
        log.warning(f"Invalid sample_rate {sample_rate}, defaulting to 40000")
        sample_rate = 40000
    epochs = max(1, min(epochs, 10000))

    # Auto-detect optimal batch_size from GPU VRAM if not specified
    if batch_size <= 0:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Optimized formula: better GPU utilization, especially for RTX 4090 (24GB)
            # RTX 4090 (24GB) → 32, RTX 3090 (24GB) → 28, RTX 3060 (12GB) → 16
            batch_size = min(48, max(4, (int(vram_gb) // 3) * 4))
            log.info(f"Auto-detected batch_size={batch_size} for {vram_gb:.1f}GB VRAM")
        except Exception:
            batch_size = 4  # GPU VRAM 감지 실패 시 보수적 기본값
    batch_size = max(4, min(48, batch_size))

    job_id = job.get("id") or uuid.uuid4().hex[:12]
    start_time = time.time()

    # Applio expects a specific directory structure for training:
    #   logs/{model_name}/       - training logs, checkpoints
    #   logs/{model_name}/1_2_3/ - preprocessed sliced audio
    #   logs/{model_name}/0_gt_wavs/ - preprocessed audio
    #   logs/{model_name}/3_feature768/ or 3_feature256/ - feature files
    logs_dir = APPLIO_ROOT / "logs" / f"{model_name}_{job_id}"
    dataset_dir = ensure_dir(WORK_DIR / f"train_{job_id}" / "dataset")
    ensure_dir(logs_dir)

    # Map sample rate to Applio's internal version string
    sr_map = {32000: "32k", 40000: "40k", 48000: "48k"}
    sr_label = sr_map.get(sample_rate, "40k")

    try:
        # --- Step 1: Download/decode audio files to dataset directory ---
        total_files = len(audio_files) + len(audio_urls)
        runpod.serverless.progress_update(job, f"Downloading {total_files} audio files... (1/5)")

        # R2 URL에서 다운로드 (네트워크/HTTP 오류 시 최대 5회 재시도)
        import requests as _req
        for i, uobj in enumerate(audio_urls):
            fname = Path(uobj.get("filename", f"audio_{i}.wav")).name  # path traversal 방지
            url = uobj["url"]
            dest = dataset_dir / fname
            last_err = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    resp = _req.get(url, timeout=120)
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        f.write(resp.content)
                    log.info(f"Downloaded training file: {fname} ({len(resp.content) / 1024:.1f} KB)")
                    last_err = None
                    break
                except _req.exceptions.HTTPError as dl_err:
                    status_code = getattr(dl_err.response, "status_code", 0) if hasattr(dl_err, "response") else 0
                    # 403/404/401 = 영구적 오류 → 재시도 무의미
                    if status_code in (401, 403, 404):
                        raise RuntimeError(
                            f"학습 파일 다운로드 실패 ({fname}, HTTP {status_code}): "
                            f"URL이 만료되었거나 파일이 존재하지 않습니다."
                        ) from dl_err
                    last_err = dl_err
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 10)
                        log.warning(f"다운로드 재시도 {attempt + 1}/{max_retries} ({fname}, HTTP {status_code}), "
                                    f"{wait}초 후: {dl_err}")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"학습 파일 다운로드 {max_retries}회 실패 ({fname}): {dl_err}"
                        ) from dl_err
                except (_req.exceptions.Timeout, _req.exceptions.ConnectionError) as dl_err:
                    last_err = dl_err
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 10)
                        log.warning(f"다운로드 재시도 {attempt + 1}/{max_retries} ({fname}), "
                                    f"{wait}초 후: {dl_err}")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"학습 파일 다운로드 {max_retries}회 실패 ({fname}): {dl_err}"
                        ) from dl_err
                except Exception as dl_err:
                    raise RuntimeError(f"학습 파일 다운로드 실패 ({fname}): {dl_err}") from dl_err
            if last_err:
                raise RuntimeError(f"학습 파일 다운로드 실패 ({fname}): {last_err}")

        # base64 인라인 디코딩 (하위 호환)
        for i, fobj in enumerate(audio_files):
            fname = fobj.get("filename", f"audio_{i}.wav")
            dest = dataset_dir / fname
            decode_b64_file(fobj["data_base64"], dest)
            log.info(f"Decoded training file: {fname}")

        # Convert all to WAV at target sample rate
        wav_dir = ensure_dir(WORK_DIR / f"train_{job_id}" / "wav")
        wav_idx = 0
        for f in sorted(dataset_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                # Use index prefix to avoid collision when stems match
                # (e.g., song.mp3 and song.flac → 0000_song.wav, 0001_song.wav)
                out_wav = wav_dir / f"{wav_idx:04d}_{f.stem}.wav"
                wav_idx += 1

                # v13: MP3 소스 감지 시 HF 아티팩트 정리 필터 추가
                # MP3 인코더(128~320kbps)는 15-16kHz 이상을 버리거나 노이즈/링잉으로 채움
                # 이 HF 노이즈를 그대로 학습하면 모델이 "음색"으로 학습 → 금속성 고음 출력
                # 해결: 15.5kHz brick-wall lowpass로 MP3 HF 인코딩 아티팩트 제거 후 변환
                # (WAV/FLAC/AIFF 등 무손실 소스는 그대로 — lowpass 불필요)
                if f.suffix.lower() in {".mp3", ".m4a", ".aac", ".wma", ".ogg"}:
                    af_chain = (
                        "highpass=f=60:poles=2,"          # 60Hz 이하 럼블 제거
                        "lowpass=f=15500:poles=2,"         # MP3 HF 아티팩트 제거 (15.5kHz 이상 차단)
                        f"aresample=resampler=soxr:precision=28:osr={sample_rate}"
                    )
                    log.info(f"MP3/lossy source detected: applying HF cleanup for {f.name}")
                else:
                    af_chain = f"aresample=resampler=soxr:precision=28:osr={sample_rate}"

                run_ffmpeg([
                    "-i", str(f),
                    "-af", af_chain,
                    "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    str(out_wav),
                ])

        # --- Step 2: Preprocess (resample to target SR) ---
        runpod.serverless.progress_update(job, "Preprocessing audio (slicing, resampling)... (2/5)")
        try:
            _rvc_preprocess(model_name, str(wav_dir), sample_rate, logs_dir)
        except Exception as e:
            raise RuntimeError(f"[2/5 전처리] {e}") from e

        # --- Step 3+4: Extract F0 + features + config + filelist ---
        runpod.serverless.progress_update(job, f"Extracting F0 & features ({f0_method}, {embedder_model})... (3/5)")
        try:
            _rvc_extract(model_name, f0_method, sample_rate, logs_dir, embedder_model)
        except Exception as e:
            raise RuntimeError(f"[3/5 추출] {e}") from e

        # --- Step 4: Train ---
        runpod.serverless.progress_update(job, f"Training model ({epochs} epochs)... (4/5)")
        train_info = _rvc_train(
            model_name=model_name,
            sample_rate=sample_rate,
            epochs=epochs,
            batch_size=batch_size,
            save_every_epoch=save_every_epoch,
            logs_dir=logs_dir,
            sr_label=sr_label,
            job=job,
            pretrained_model=pretrained_model,
        )

        # --- Step 6: Generate FAISS index ---
        runpod.serverless.progress_update(job, "Generating FAISS index... (5/5)")
        _rvc_create_index(model_name, logs_dir)

        # --- Collect output files ---
        # Diagnostic: list all files in logs_dir to understand what training produced
        if logs_dir.exists():
            all_files = list(logs_dir.rglob("*"))
            pth_files = [f for f in all_files if f.suffix == ".pth"]
            log.info(f"Logs dir contents: {len(all_files)} total files, "
                     f"{len(pth_files)} .pth files")
            for pf in pth_files:
                log.info(f"  Found .pth: {pf} ({pf.stat().st_size / 1024 / 1024:.1f} MB)")
            if not pth_files:
                # Log directory tree for debugging
                dirs = sorted(set(f.parent for f in all_files))
                log.error(f"No .pth files found! Directory tree: {[str(d) for d in dirs]}")
                # Also list actual files for more detail
                for af in sorted(all_files):
                    if af.is_file():
                        log.error(f"  File: {af} ({af.stat().st_size:,} bytes)")
        else:
            log.error(f"Logs directory does not exist: {logs_dir}")

        pth_path = _find_best_pth(logs_dir, model_name)
        index_path = _find_index(logs_dir, model_name)

        if pth_path is None:
            train_tail = "\n".join(train_info.get("last_lines", []))
            epoch_count = train_info.get("epoch_count", 0)
            raise RuntimeError(
                f"Training completed but no .pth model found in {logs_dir}. "
                f"Epochs detected: {epoch_count}/{epochs}. "
                f"Training subprocess output (last 15 lines):\n{train_tail}"
            )

        elapsed = time.time() - start_time

        # Upload model files to S3-compatible storage via RunPod's rp_upload.
        # RunPod payload limit is ~20 MB — RVC models are ~52 MB, so inline
        # base64 is not viable. Bucket upload returns a 7-day presigned URL.
        pth_size_mb = pth_path.stat().st_size / 1024 / 1024
        log.info(f"Uploading model: {pth_path.name} ({pth_size_mb:.1f} MB)")

        # v36: 모든 체크포인트 목록 수집 (에폭별 비교 청취용)
        _all_checkpoints = []
        for _ckpt_loc in [logs_dir, logs_dir / "weights",
                          APPLIO_ROOT / "logs" / model_name / "weights"]:
            if _ckpt_loc.exists():
                for _ckpt in _ckpt_loc.glob("*.pth"):
                    if not _ckpt.stem.startswith(("G_", "D_")):
                        _all_checkpoints.append({
                            "filename": _ckpt.name,
                            "size_mb": round(_ckpt.stat().st_size / 1024 / 1024, 1),
                            "is_best": "best_epoch" in _ckpt.name,
                        })
        _all_checkpoints.sort(key=lambda x: x["filename"])
        log.info(f"Found {len(_all_checkpoints)} epoch checkpoints for comparison")

        result = {
            "model_name": model_name,
            "epochs_trained": epochs,
            "sample_rate": sample_rate,
            "training_time_seconds": round(elapsed, 1),
            "pth_filename": pth_path.name,
            "checkpoints": _all_checkpoints,  # v36: 에폭별 비교 청취용
        }

        # --- Upload strategy: R2 bucket → base64 fallback ---
        upload_ok = False
        try:
            from runpod.serverless.utils import upload_file_to_bucket

            bucket_name = job_input.get("bucket_name") or os.environ.get("BUCKET_NAME", "")

            pth_url = upload_file_to_bucket(
                file_name=pth_path.name,
                file_location=str(pth_path),
                prefix=f"train/{job_id}",
                bucket_name=bucket_name,
            )
            if pth_url and pth_url.startswith("http"):
                result["pth_url"] = pth_url
                upload_ok = True
                log.info(f"Model uploaded to bucket: {pth_url[:80]}...")

                if index_path is not None:
                    index_url = upload_file_to_bucket(
                        file_name=index_path.name,
                        file_location=str(index_path),
                        prefix=f"train/{job_id}",
                        bucket_name=bucket_name,
                    )
                    if index_url and index_url.startswith("http"):
                        result["index_url"] = index_url
                        result["index_filename"] = index_path.name
                        log.info(f"Index uploaded to bucket: {index_url[:80]}...")
                else:
                    log.warning("No FAISS index file generated")
            else:
                log.warning("Bucket upload returned non-HTTP path, falling back to base64")

        except ImportError:
            log.warning("runpod upload utils not available, falling back to base64")
        except Exception as e:
            log.warning(f"Bucket upload failed ({e}), falling back to base64")

        # Fallback: base64 encode model if bucket upload failed
        # RunPod payload limit ~20MB, RVC models ~50MB → only works for small models
        # base64 expands size by 4/3, so effective raw-file limit is ~13.5MB (13.5 * 4/3 = 18MB)
        if not upload_ok:
            import base64
            pth_size_mb = pth_path.stat().st_size / (1024 * 1024)
            pth_b64_mb = pth_size_mb * 4 / 3  # base64 overhead
            if pth_b64_mb <= 18:
                log.info(f"Using base64 fallback for {pth_path.name} ({pth_size_mb:.1f} MB raw / {pth_b64_mb:.1f} MB b64)")
                with open(pth_path, "rb") as f:
                    result["pth_base64"] = base64.b64encode(f.read()).decode()
                result["upload_method"] = "base64"

                if index_path is not None:
                    idx_size_mb = index_path.stat().st_size / (1024 * 1024)
                    remaining_mb = (18 - pth_b64_mb) * 3 / 4  # convert back to raw MB
                    if idx_size_mb <= remaining_mb:
                        with open(index_path, "rb") as f:
                            result["index_base64"] = base64.b64encode(f.read()).decode()
                        result["index_filename"] = index_path.name
                        log.info(f"Index included via base64 ({idx_size_mb:.1f} MB)")
                    else:
                        log.warning(f"Index too large for base64 ({idx_size_mb:.1f} MB), skipped")
            else:
                log.error(f"Model too large for base64 ({pth_size_mb:.1f} MB raw / {pth_b64_mb:.1f} MB b64) and bucket upload failed")
                result["upload_method"] = "failed"
                result["error"] = (
                    "모델 파일이 너무 크고 클라우드 스토리지가 설정되지 않았습니다. "
                    "RunPod 템플릿에 BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, "
                    "BUCKET_SECRET_ACCESS_KEY 환경변수를 설정하세요."
                )

        return result

    finally:
        try:
            cleanup_dir(WORK_DIR / f"train_{job_id}")
            # logs_dir is now job-specific (includes job_id), so it's safe to remove entirely.
            # The model/index have already been uploaded/encoded above.
            if logs_dir and logs_dir.exists():
                try:
                    shutil.rmtree(str(logs_dir), ignore_errors=True)
                    log.info(f"Cleaned up training logs: {logs_dir.name}")
                except Exception:
                    pass
        finally:
            cleanup_gpu()


def _rvc_preprocess(
    model_name: str, dataset_path: str, sample_rate: int, logs_dir: Path
) -> None:
    """
    RVC preprocessing v35: Demucs vocal separation + slice + filter + resample.

    Self-contained implementation using FFmpeg — avoids Applio's import chain.

    Steps:
      0) [v35 NEW] Demucs 보컬 분리 — MR(반주) 제거 (학습 소스에 MR 포함 시 기계음 원인)
      1) Slice each audio file into ~5.0s segments (optimized for singing voice)
      2) Quality filter: RMS(-45dBFS) + spectral flatness(0.45) + SNR(10dB) check
      3) Resample segments to target sample rate → sliced_audios/
      4) Resample segments to 16kHz → sliced_audios_16k/
      5) High-note oversampling: F0≥350Hz 세그먼트 복제 (고음 비율 2%→10%+)
      6) Pitch-shift augmentation: 중고음 세그먼트 +2/+4 semitone 복사
      7) Global normalization: -1 dBFS (자연 다이나믹스 보존)

    v11 변경:
      - 스펙트럴 품질 필터 추가 (노이즈 세그먼트 자동 제거)
      - 고음 오버샘플링 (F0 분석 기반, 고음 학습 데이터 비율 증가)
      - 피치 시프트 증강 (중고음 → +2/+4 semitone → 가상 고음역 데이터 생성)
      - 정규화 타겟 -3→-1 dBFS (v24: 자연 다이나믹스 보존)
    """
    SLICE_DURATION = 5.0  # seconds — v13: 3.5→5.0 (한국 가요 프레이즈 보존: 비브라토, 멜리스마, 호흡 패턴)

    gt_dir = ensure_dir(logs_dir / "sliced_audios")
    sr16k_dir = ensure_dir(logs_dir / "sliced_audios_16k")
    tmp_slices = ensure_dir(logs_dir / "_tmp_slices")

    dataset = Path(dataset_path)
    audio_files = sorted(
        f for f in dataset.iterdir()
        if f.suffix.lower() in AUDIO_EXTS
    )

    if not audio_files:
        raise RuntimeError(f"전처리할 오디오 파일이 없습니다: {dataset_path}")

    # ━━━ v35: Demucs 보컬 분리 — MR 제거 (학습 품질 핵심) ━━━
    # 분석 결과: 학습 소스 8개 중 6개에 MR(반주) 포함 → 기계음의 근본 원인
    # Demucs로 보컬만 추출하여 깨끗한 보컬 데이터로 학습
    vocal_dir = ensure_dir(logs_dir / "_vocal_separated")
    log.info(f"Running Demucs vocal separation on {len(audio_files)} training files...")
    try:
        separation = _demucs_separate(audio_files, vocal_dir)
        if separation.get("vocals"):
            vocal_files = [Path(p) for p in separation["vocals"]]
            log.info(f"Demucs separated {len(vocal_files)} vocal tracks from {len(audio_files)} sources")
            audio_files = vocal_files  # 이후 전처리는 분리된 보컬만 사용
        else:
            log.warning("Demucs produced no vocals, using original audio files")
    except Exception as e:
        log.warning(f"Demucs separation failed ({e}), using original audio files")

    import soundfile as _sf

    idx = 0
    quality_skipped = 0
    for audio_file in audio_files:
        try:
            duration = get_audio_duration(audio_file)
            if duration < 0.5:
                log.warning(f"Skipping too-short file ({duration:.1f}s): {audio_file.name}")
                continue

            if duration <= SLICE_DURATION * 1.5:
                slice_paths = [audio_file]
            else:
                slice_prefix = tmp_slices / f"s{idx:04d}"
                slice_pattern = f"{slice_prefix}_%05d.wav"
                run_ffmpeg([
                    "-i", str(audio_file),
                    "-f", "segment",
                    "-segment_time", str(SLICE_DURATION),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate),
                    slice_pattern,
                ])
                slice_paths = sorted(tmp_slices.glob(f"s{idx:04d}_*.wav"))
                if not slice_paths:
                    log.warning(f"Slicing produced 0 segments for {audio_file.name}")
                    continue

            for sp in slice_paths:
                seg_dur = get_audio_duration(sp)
                if seg_dur < 0.3:
                    continue

                padded = f"0_{idx:07d}"

                gt_path = gt_dir / f"{padded}.wav"
                run_ffmpeg([
                    "-i", str(sp),
                    "-af", (
                        "highpass=f=50:poles=2,"
                        f"aresample=resampler=soxr:precision=28:osr={sample_rate}"
                    ),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(gt_path),
                ])

                _audio, _sr = _sf.read(str(gt_path))

                # ── 품질 필터 1: 조용한 세그먼트 ──
                _rms_db = 20 * np.log10(np.sqrt(np.mean(_audio ** 2)) + 1e-10)
                if _rms_db < -45:
                    gt_path.unlink(missing_ok=True)
                    log.info(f"Skipped quiet segment ({_rms_db:.1f}dB): {padded}")
                    continue

                # ── 품질 필터 2: 스펙트럴 평탄도 (v11 신규) ──
                # 평탄도 > 0.45 → 노이즈에 가까운 세그먼트 (정상 음성은 0.01-0.2)
                # v13: 0.4→0.45 완화 — MP3 소스는 HF 노이즈로 인해 flatness가 다소 높게 측정
                # 0.4 임계값으로는 유효한 MP3 보컬 세그먼트까지 제거될 수 있음
                _S = np.abs(np.fft.rfft(_audio))
                _S_power = _S ** 2 + 1e-20
                _geo = np.exp(np.mean(np.log(_S_power)))
                _arith = np.mean(_S_power)
                _flatness = _geo / (_arith + 1e-20)
                if _flatness > 0.45:
                    gt_path.unlink(missing_ok=True)
                    quality_skipped += 1
                    log.info(f"Skipped noisy segment (flatness={_flatness:.3f}): {padded}")
                    continue

                # ── 품질 필터 3: SNR 추정 (v11 신규) ──
                # 20ms 프레임 기반 SNR — 최하위 10% 프레임을 노이즈 기준으로 추정
                _frame_sz = max(1, int(_sr * 0.02))
                _n_fr = max(1, len(_audio) // _frame_sz)
                _fr_rms = np.array([
                    np.sqrt(np.mean(_audio[i * _frame_sz:(i + 1) * _frame_sz] ** 2))
                    for i in range(_n_fr)
                ])
                _noise_rms = float(np.mean(np.sort(_fr_rms)[:max(1, _n_fr // 10)]))
                _sig_rms = float(np.sqrt(np.mean(_audio ** 2)))
                _snr = 20 * np.log10(max(_sig_rms, 1e-10) / max(_noise_rms, 1e-10))
                if _snr < 10:
                    gt_path.unlink(missing_ok=True)
                    quality_skipped += 1
                    log.info(f"Skipped low-SNR segment (SNR={_snr:.1f}dB): {padded}")
                    continue

                # 16kHz 버전
                sr16k_path = sr16k_dir / f"{padded}.wav"
                run_ffmpeg([
                    "-i", str(sp),
                    "-af", (
                        "highpass=f=50:poles=2,"
                        "aresample=resampler=soxr:precision=28:osr=16000"
                    ),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(sr16k_path),
                ])

                idx += 1

        except Exception as e:
            log.warning(f"Failed to preprocess {audio_file.name}: {e}")
            continue

    cleanup_dir(tmp_slices)

    if idx == 0:
        raise RuntimeError("전처리에 성공한 오디오 파일이 없습니다")

    if quality_skipped > 0:
        log.info(f"Quality filter: {quality_skipped} noisy/low-SNR segments skipped")

    original_count = idx

    # ══════════════════════════════════════════════════════════════════════════
    # v11: 고음 오버샘플링 — F0 분석 기반으로 고음 세그먼트 복제
    # 학습 데이터의 고음(C5+) 비율이 2.3%로 극도로 부족 → 오버샘플링으로 10%+ 확보
    # ══════════════════════════════════════════════════════════════════════════
    _gt_originals = sorted(gt_dir.glob("*.wav"))
    _oversampled = 0
    _os_idx = idx

    try:
        import librosa as _lr

        log.info(f"Analyzing pitch of {len(_gt_originals)} segments for oversampling...")
        for _gf in _gt_originals:
            try:
                _a, _sr_a = _sf.read(str(_gf))
                _f0, _, _ = _lr.pyin(
                    _a.astype(np.float32), fmin=80, fmax=1200,
                    sr=_sr_a, frame_length=2048,
                )
                _f0_v = _f0[~np.isnan(_f0)]
                if len(_f0_v) < 3:
                    continue
                _f0_med = float(np.median(_f0_v))

                # 복제 횟수 결정: 고음일수록 더 많이 복제
                # v13: 오버샘플링 추가 축소 (v12: 2/1/0 → v13: 1/0/0)
                # 이유: MP3 소스처럼 아티팩트 있는 데이터는 오버샘플링 시
                #       같은 아티팩트가 반복 학습 → 기계음 강화. 1회 복제로 최소화.
                if _f0_med >= 440:      # A4+ → 1배 복제만 (C5 이상 고음 약간 보강)
                    copies = 1
                else:
                    copies = 0          # 그 외 → 복제 없음

                for _ in range(copies):
                    _pad = f"0_{_os_idx:07d}"
                    _dst_gt = gt_dir / f"{_pad}.wav"
                    _dst_16k = sr16k_dir / f"{_pad}.wav"
                    shutil.copy2(str(_gf), str(_dst_gt))
                    _src_16k = sr16k_dir / _gf.name
                    if _src_16k.exists():
                        shutil.copy2(str(_src_16k), str(_dst_16k))
                    _os_idx += 1
                    _oversampled += 1
            except Exception:
                continue

        if _oversampled > 0:
            log.info(f"High-note oversampling: +{_oversampled} copies added "
                     f"({original_count}→{_os_idx} segments)")
    except ImportError:
        log.warning("librosa not available, skipping high-note oversampling")
    except Exception as _ose:
        log.warning(f"Oversampling failed: {_ose}")

    # ══════════════════════════════════════════════════════════════════════════
    # v13: 피치 시프트 증강 완전 제거
    # 이전(v11/v12): asetrate 기반 +2 semitone 증강
    # 문제: asetrate는 재생 속도를 변경한 뒤 리샘플 → 피치가 올라가지만 템포도 12% 빨라짐
    #       즉 "빠른 노래"로 변형된 데이터를 "같은 노래의 고음"으로 학습 → 기계음 원인
    #       모델이 "고음 = 빠른 발음"으로 잘못 학습 → 고음 구간에서 기계적 급격한 피치 변화
    # 결론: asetrate 증강은 품질 개선이 아니라 기계음의 직접 원인 — 완전 제거
    # 고음 데이터 부족 문제는 위의 oversampling(F0 기반 복제)으로 해결
    # ══════════════════════════════════════════════════════════════════════════
    _augmented = 0
    _aug_idx = _os_idx
    log.info("Pitch-shift augmentation: DISABLED (v13 — asetrate tempo-distortion 제거)")

    total_segments = len(list(gt_dir.glob("*.wav")))
    log.info(f"Pre-normalization total: {total_segments} segments "
             f"(original={original_count}, oversampled=+{_oversampled}, augmented=+{_augmented})")

    # ── 글로벌 정규화 ──────────────────────────────────────────────────────────
    # 전체 학습셋(원본 + 오버샘플 + 증강)의 최대 피크 기준으로 동일 계수 적용
    # → 슬라이스 간 상대적 볼륨 관계 보존 → 자연스러운 다이나믹스 학습
    # v11: -3→-4 dBFS (피치 시프트 증강 데이터의 피크 헤드룸 확보)
    _gt_files = sorted(gt_dir.glob("*.wav"))
    if _gt_files:
        _global_peak = 0.0
        for _gf in _gt_files:
            try:
                _a, _ = _sf.read(str(_gf))
                _p = float(np.max(np.abs(_a)))
                if _p > _global_peak:
                    _global_peak = _p
            except Exception:
                continue
        if _global_peak > 0:
            _TARGET = 10.0 ** (-1 / 20)  # -1 dBFS = 0.891 (v24: -4→-1, 자연 다이나믹스 최대 보존)
            _norm_factor = _TARGET / _global_peak
            log.info(
                f"Global normalization: peak={_global_peak:.4f}, "
                f"factor={_norm_factor:.4f}, target=-1dBFS ({len(_gt_files)} slices)"
            )
            for _gf in _gt_files:
                try:
                    _a, _sr_g = _sf.read(str(_gf))
                    _sf.write(str(_gf), _a * _norm_factor, samplerate=_sr_g, subtype="PCM_16")
                except Exception as _ne:
                    log.warning(f"Normalization failed for {_gf.name}: {_ne}")
            for _gf16 in sorted(sr16k_dir.glob("*.wav")):
                try:
                    _a16, _sr16_g = _sf.read(str(_gf16))
                    if np.any(np.isnan(_a16)) or np.any(np.isinf(_a16)):
                        log.warning(f"Corrupted 16k file {_gf16.name}, deleting (+ matching gt)")
                        _gf16.unlink(missing_ok=True)
                        _matching_gt = gt_dir / _gf16.name
                        if _matching_gt.exists():
                            _matching_gt.unlink(missing_ok=True)
                        continue
                    _sf.write(str(_gf16), _a16 * _norm_factor, samplerate=_sr16_g, subtype="PCM_16")
                except Exception as _ne16:
                    log.warning(f"Normalization failed for 16k {_gf16.name}: {_ne16}")
    # ──────────────────────────────────────────────────────────────────────────

    final_count = len(list(gt_dir.glob("*.wav")))
    log.info(f"RVC preprocess v11 completed: {final_count} total segments "
             f"(orig={original_count}, oversample=+{_oversampled}, augment=+{_augmented}) "
             f"→ {gt_dir} + {sr16k_dir}")

    # Create config.json from Applio's sample-rate template
    config_src = APPLIO_ROOT / "rvc" / "configs" / f"{sample_rate}.json"
    config_dst = logs_dir / "config.json"
    if config_src.exists() and not config_dst.exists():
        shutil.copyfile(str(config_src), str(config_dst))
        log.info(f"Created config.json from {config_src.name}")
    elif not config_src.exists():
        log.warning(f"Config template not found: {config_src}")


def _rvc_extract(
    model_name: str, f0_method: str, sample_rate: int, logs_dir: Path,
    embedder_model: str = "contentvec",
) -> None:
    """
    Extract F0 + features + generate config.json + filelist.txt.

    Calls rvc/train/extract/extract.py DIRECTLY (bypasses core.py which
    swallows subprocess failures). extract.py does everything in one call:
    F0 extraction, feature extraction, config generation, and filelist creation.

    Positional args for extract.py:
      1: model_path (logs dir), 2: f0_method, 3: cpu_cores, 4: gpu,
      5: sample_rate, 6: embedder_model, 7: embedder_model_custom,
      8: include_mutes
    """
    cmd = [
        sys.executable,
        str(APPLIO_ROOT / "rvc" / "train" / "extract" / "extract.py"),
        str(logs_dir),          # 1: model_path (experiment dir)
        f0_method,              # 2: f0_method
        str(os.cpu_count() or 4),  # 3: cpu_cores
        "0",                    # 4: gpu
        str(sample_rate),       # 5: sample_rate
        embedder_model,         # 6: embedder_model
        "None",                 # 7: embedder_model_custom
        "2",                    # 8: include_mutes
    ]

    log.info(f"Starting extraction: method={f0_method}, sr={sample_rate}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(APPLIO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_lines = []
    extract_timeout = 7200  # 2 hours

    def _drain_stdout():
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            log.info(f"[extract] {line}")
            last_lines.append(line)
            if len(last_lines) > 20:
                last_lines.pop(0)

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()
    reader.join(timeout=extract_timeout)

    if reader.is_alive():
        proc.kill()
        proc.wait(timeout=10)
        raise RuntimeError("Feature extraction 타임아웃 (2시간 초과). 학습 데이터가 너무 많을 수 있습니다.")

    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise RuntimeError("Feature extraction 타임아웃 (stdout 종료 후 프로세스 미종료)")
    if proc.returncode != 0:
        tail = "\n".join(last_lines[-10:])
        raise RuntimeError(
            f"Extraction failed (exit code {proc.returncode}).\n"
            f"Last output:\n{tail}"
        )

    # Verify critical outputs exist
    extracted_dir = logs_dir / "extracted"
    f0_dir = logs_dir / "f0"
    config_json = logs_dir / "config.json"
    filelist_txt = logs_dir / "filelist.txt"

    # Check feature extraction (most common silent failure point)
    if not extracted_dir.exists() or not any(extracted_dir.glob("*.npy")):
        npy_count = len(list(extracted_dir.glob("*.npy"))) if extracted_dir.exists() else 0
        raise RuntimeError(
            f"Feature extraction failed: extracted/ has {npy_count} .npy files. "
            f"This usually means the embedder model ({embedder_model}) failed to load. "
            f"Check that transformers is compatible with the installed PyTorch version."
        )
    else:
        npy_count = len(list(extracted_dir.glob("*.npy")))
        log.info(f"Feature extraction OK: {npy_count} .npy files in extracted/")

    # Check F0 extraction
    if not f0_dir.exists() or not any(f0_dir.glob("*.npy")):
        f0_count = len(list(f0_dir.glob("*.npy"))) if f0_dir.exists() else 0
        raise RuntimeError(
            f"F0 extraction failed: f0/ has {f0_count} .npy files. "
            f"Method: {f0_method}"
        )

    if not config_json.exists():
        log.warning("extract.py did not create config.json, creating from template")
        config_src = APPLIO_ROOT / "rvc" / "configs" / f"{sample_rate}.json"
        if config_src.exists():
            shutil.copyfile(str(config_src), str(config_json))
    if not filelist_txt.exists():
        log.warning("extract.py did not create filelist.txt — training may fail")
    else:
        # Verify filelist has actual training data (not just mute entries)
        with open(filelist_txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip() and "mute" not in ln.lower()]
        log.info(f"filelist.txt: {len(lines)} training entries (excluding mutes)")
        if len(lines) == 0:
            raise RuntimeError(
                "filelist.txt has no training entries (only mute placeholders). "
                "Feature or F0 extraction may have produced mismatched files."
            )

    log.info(f"Extraction verified: features={npy_count}, config={'OK' if config_json.exists() else 'MISSING'}, filelist={'OK' if filelist_txt.exists() else 'MISSING'}")



def _rvc_train(
    model_name: str,
    sample_rate: int,
    epochs: int,
    batch_size: int,
    save_every_epoch: int,
    logs_dir: Path,
    sr_label: str,
    job: dict,
    pretrained_model: str = "klm49",
) -> dict:
    """
    Run RVC v2 training loop via Applio CLI.
    Uses CLI (subprocess) instead of core API because core API
    doesn't check subprocess return codes.
    """
    # Determine pretrained model paths based on user selection
    pretrained_g = _find_pretrained("G", sr_label, pretrained_model)
    pretrained_d = _find_pretrained("D", sr_label, pretrained_model)

    # Log training parameters for debugging
    log.info(f"Training params: model={model_name}, sr={sample_rate}, epochs={epochs}, "
             f"batch={batch_size}, save_every={save_every_epoch}, sr_label={sr_label}")
    log.info(f"Pretrained G: {pretrained_g}")
    log.info(f"Pretrained D: {pretrained_d}")
    if not pretrained_g or not pretrained_d:
        log.warning(
            "⚠ Pretrained model(s) NOT found — training from random initialization! "
            "Quality will be significantly worse. Ensure pretrained_v2 models are cached."
        )
        runpod.serverless.progress_update(job, "⚠ 사전학습 모델 없이 학습 시작 (품질 저하 가능)")

    # Verify preprocessed data exists before training
    gt_dir = logs_dir / "sliced_audios"
    if gt_dir.exists():
        gt_count = len(list(gt_dir.glob("*.wav")))
        log.info(f"Preprocessed audio files: {gt_count} in {gt_dir}")
        if gt_count == 0:
            raise RuntimeError("No preprocessed audio files found — cannot train")
    else:
        raise RuntimeError(f"Preprocessed directory missing: {gt_dir}")

    # Ensure batch_size doesn't exceed dataset size (causes silent training failure)
    if batch_size > gt_count:
        old_bs = batch_size
        batch_size = max(2, gt_count)
        log.warning(f"batch_size {old_bs} > dataset size {gt_count}, "
                    f"reduced to {batch_size}")
    # 과적합 경고: 데이터셋이 batch_size의 4배 미만이면 일반화 부족 가능성
    if gt_count < batch_size * 4:
        log.warning(f"Small dataset ({gt_count} samples) relative to batch_size={batch_size}. "
                    f"Consider using batch_size={max(2, gt_count // 4)} to avoid overfitting.")

    # Also verify filelist.txt exists and has entries
    filelist_txt = logs_dir / "filelist.txt"
    if filelist_txt.exists():
        with open(filelist_txt, "r") as f:
            fl_lines = [ln.strip() for ln in f if ln.strip() and "mute" not in ln.lower()]
        log.info(f"filelist.txt before training: {len(fl_lines)} entries")
        if not fl_lines:
            raise RuntimeError(
                "filelist.txt is empty (no training data entries). "
                "Check that preprocessing and extraction produced matching files."
            )
    else:
        raise RuntimeError(
            f"filelist.txt not found at {filelist_txt}. "
            "Preprocessing or extraction did not produce training data. "
            "Check that audio files were properly preprocessed."
        )

    # Call train.py DIRECTLY — bypassing core.py which swallows subprocess errors.
    # train.py uses positional sys.argv arguments in this exact order:
    #   1: model_name, 2: save_every_epoch, 3: total_epoch,
    #   4: pretrainG, 5: pretrainD, 6: gpu, 7: batch_size, 8: sample_rate,
    #   9: save_only_latest, 10: save_every_weights, 11: cache_data_in_gpu,
    #   12: overtraining_detector, 13: overtraining_threshold, 14: cleanup,
    #   15: vocoder, 16: checkpointing
    pg = str(pretrained_g) if pretrained_g else ""
    pd = str(pretrained_d) if pretrained_d else ""
    # train.py constructs paths as logs/{arg1}/config.json, so arg1 must match
    # the actual directory name (which includes job_id for collision avoidance).
    train_dir_name = logs_dir.name
    cmd = [
        sys.executable,
        str(APPLIO_ROOT / "rvc" / "train" / "train.py"),
        train_dir_name,           # 1: must match logs/ subdirectory name
        str(save_every_epoch),    # 2
        str(epochs),              # 3
        pg,                       # 4: pretrainG path
        pd,                       # 5: pretrainD path
        "0",                      # 6: gpu
        str(batch_size),          # 7
        str(sample_rate),         # 8
        "True",                   # 9: save_only_latest (avoid G_/D_ checkpoint bloat)
        "True",                   # 10: save_every_weights
        "True",                   # 11: cache_data_in_gpu (RTX 4090 24GB VRAM → GPU 캐시로 훈련 속도 2× 향상)
        "True",                   # 12: overtraining_detector
        "25",                     # 13: overtraining_threshold (v45: 50→25 KLM 권장 — 소량 데이터에서 과적합 조기 감지)
        "False",                  # 14: cleanup
        "HiFi-GAN",              # 15: vocoder
        "False",                  # 16: checkpointing
    ]

    log.info(f"Starting training: {epochs} epochs, cmd={' '.join(cmd[:4])}...")
    log.info(f"Full train args: {cmd[2:]}")

    # CUDA_LAUNCH_BLOCKING=1 for synchronous CUDA errors (better diagnostics)
    train_env = os.environ.copy()
    train_env["CUDA_LAUNCH_BLOCKING"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(APPLIO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=train_env,
    )

    # Stream output and report progress
    epoch_count = 0
    last_lines: list[str] = []
    child_error: str | None = None
    for line in iter(proc.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue
        log.info(f"[train] {line}")
        last_lines.append(line)
        if len(last_lines) > 30:
            last_lines.pop(0)

        # Detect child process crash (train.py spawns multiprocessing.Process
        # which can crash without setting a non-zero exit code on the parent)
        if any(err in line for err in ("Error", "Exception", "Traceback")):
            if "Traceback" in line:
                child_error = ""  # start capturing
            elif child_error is not None:
                child_error += line + "\n"
        if child_error and ("Error:" in line or "Exception:" in line):
            child_error += line
            # Keep it but don't raise yet — collect more context

        # Parse epoch progress from training output.
        # Applio outputs: "model | epoch=394 | step=14578 | ..."
        # Also tqdm batch bars: "35%|███▌ | 13/37 [00:03<00:05, 4.49it/s]"
        # We must distinguish epoch lines from tqdm batch bars.
        prev_epoch = epoch_count
        epoch_match = re.search(r'epoch[=:\s]+(\d+)', line, re.IGNORECASE)
        if epoch_match:
            parsed_epoch = int(epoch_match.group(1))
            if parsed_epoch > epoch_count:
                epoch_count = parsed_epoch
        elif ("Epoch" in line or "epoch" in line) and "/" in line and "it/s" not in line:
            # Fallback: "Epoch 50/300" format (NOT tqdm batch bars)
            try:
                for part in line.replace(":", " ").split():
                    if "/" in part:
                        current, total = part.split("/")[:2]
                        current = current.strip().lstrip("|").strip()
                        if current.isdigit() and total.strip().isdigit():
                            parsed_epoch = int(current)
                            if parsed_epoch > epoch_count:
                                epoch_count = parsed_epoch
                            break
            except (ValueError, IndexError):
                pass

        # Only send progress_update when epoch actually advances (avoid spam)
        if epoch_count > prev_epoch:
            pct = min(95, int((epoch_count / epochs) * 100))
            try:
                runpod.serverless.progress_update(
                    job, f"Training epoch {epoch_count}/{epochs} ({pct}%)"
                )
            except Exception:
                pass

    try:
        proc.wait(timeout=36000)  # 10-hour timeout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise RuntimeError(
            f"학습 타임아웃 (10시간 초과). epoch={epoch_count}/{epochs} 에서 중단됨."
        )
    if proc.returncode != 0:
        tail = "\n".join(last_lines[-10:])
        raise RuntimeError(
            f"학습 프로세스 실패 (exit code {proc.returncode}).\n"
            f"마지막 출력:\n{tail}"
        )

    # train.py uses multiprocessing.Process — the child can crash
    # while the parent exits 0. Check for captured errors.
    tail = "\n".join(last_lines[-15:])
    if child_error and epoch_count == 0:
        raise RuntimeError(
            f"학습 자식 프로세스가 에포크 시작 전 크래시. "
            f"에러:\n{child_error}\n"
            f"마지막 출력:\n{tail}"
        )

    # If training exited with 0 but no epochs detected, the training loop
    # likely never ran (multiprocessing child crash, CUDA error, empty dataset).
    if epoch_count == 0:
        raise RuntimeError(
            f"학습 프로세스가 에포크 0에서 종료됨 (train.py가 multiprocessing 자식에서 크래시했을 가능성). "
            f"마지막 출력:\n{tail}"
        )

    log.info(f"Training completed via CLI ({epochs} epochs, reached epoch {epoch_count})")

    return {"epoch_count": epoch_count, "last_lines": last_lines[-15:]}


def _find_pretrained(gen_or_disc: str, sr_label: str, pretrained_model: str = "klm49") -> Optional[Path]:
    """
    Find pretrained generator/discriminator model for transfer learning.
    pretrained_model: "klm49" (한국어) or "rin_e3" (다국어/팝송)
    """
    # pretrained_model에 따라 우선 검색 디렉토리 결정
    _PRETRAINED_DIRS = {
        "klm49": "pretrained_v2",       # KLM49_HFG 한국어 (기본 경로)
        "rin_e3": "pretrained_rin_e3",   # RIN_E3 다국어/범용
    }
    primary_dir = _PRETRAINED_DIRS.get(pretrained_model, "pretrained_v2")

    search_dirs = [
        PRETRAINED_DIR / primary_dir,           # 사용자 선택 모델 우선
        PRETRAINED_DIR / "pretrained_v2",       # 폴백: KLM49 (기본)
        PRETRAINED_DIR,
        APPLIO_ROOT / "assets" / "pretrained_v2",
        APPLIO_ROOT / "rvc" / "pretraineds" / "pretrained_v2",
    ]

    patterns = [
        f"f0{gen_or_disc}{sr_label}.pth",
        f"{gen_or_disc}{sr_label}.pth",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidate = search_dir / pattern
            if candidate.exists():
                log.info(f"Found pretrained {gen_or_disc} ({pretrained_model}): {candidate}")
                return candidate

    log.warning(f"No pretrained {gen_or_disc} model found for {sr_label} (type={pretrained_model})")
    return None


def _rvc_create_index(model_name: str, logs_dir: Path) -> None:
    """
    Create FAISS index from extracted features for inference-time retrieval.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(APPLIO_ROOT)
        from core import run_index_script

        run_index_script(
            model_name=logs_dir.name,
            index_algorithm="Auto",
        )
        log.info("FAISS index created via core API")
        return
    except Exception as e:
        log.warning(f"Index creation via core API failed: {e}")
    finally:
        os.chdir(original_cwd)

    try:
        import faiss

        # Applio extract.py may store features in different directories
        # depending on version: extracted/, 3_feature768/, or 3_feature256/
        feature_dir = None
        for candidate_name in ("extracted", "3_feature768", "3_feature256"):
            candidate = logs_dir / candidate_name
            if candidate.exists() and any(candidate.glob("*.npy")):
                feature_dir = candidate
                break

        if feature_dir is None:
            log.warning(f"Feature directory not found in {logs_dir}")
            return

        # Collect all feature .npy files
        npys = sorted(feature_dir.glob("*.npy"))
        if not npys:
            log.warning("No .npy feature files found for index creation")
            return

        # Stack all feature vectors
        features = []
        for npy_file in npys:
            try:
                feat = np.load(str(npy_file))
                if feat.size > 0:
                    features.append(feat)
            except Exception as _npy_err:
                log.warning(f"Failed to load {npy_file.name}: {_npy_err}, skipping")
        if not features:
            log.warning("No valid feature data loaded — skipping FAISS index creation")
            return
        big_npy = np.concatenate(features, axis=0).astype(np.float32)

        if big_npy.shape[0] == 0:
            log.warning("All feature files were empty — skipping FAISS index creation")
            return

        log.info(f"Building FAISS index from {len(npys)} files, {big_npy.shape[0]} vectors")

        # Save concatenated features
        big_npy_path = logs_dir / "total_fea.npy"
        np.save(str(big_npy_path), big_npy)

        # Build IVF index (or Flat if too few vectors for IVF)
        n_vectors = big_npy.shape[0]
        dim = big_npy.shape[1]  # 768 for ContentVec, 256 for HuBERT

        if n_vectors < 40:
            # IVF는 최소 n_ivf 개의 학습 벡터가 필요 → 소규모 데이터는 Flat 인덱스 사용
            log.info(f"Small dataset ({n_vectors} vectors), using Flat index")
            index = faiss.IndexFlatL2(dim)
            index.add(big_npy)
        else:
            n_ivf = min(int(n_vectors ** 0.5), n_vectors // 2)
            n_ivf = max(1, n_ivf)
            index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
            index.train(big_npy)
            index.add(big_npy)

        index_path = logs_dir / f"{model_name}.index"
        faiss.write_index(index, str(index_path))
        log.info(f"FAISS index created: {index_path} ({big_npy.shape[0]} vectors)")

        # Validate that the index is not empty
        if hasattr(index, 'ntotal') and index.ntotal == 0:
            log.warning("FAISS index is empty — index file may not improve inference")

    except ImportError:
        log.warning("faiss not available, skipping index creation")
    except Exception as e:
        log.error(f"FAISS index creation failed: {e}")


def _find_best_pth(logs_dir: Path, model_name: str) -> Optional[Path]:
    """
    Find the best .pth voice model file from training output.

    Applio produces several types of .pth files:
      - G_*.pth / D_*.pth  → Generator/Discriminator checkpoints (400-800 MB each)
                              These are internal training state, NOT usable for inference.
      - {model_name}_*e_*s.pth → Exported voice model weights (~50 MB)
      - {model_name}_*_best_epoch.pth → Best epoch selected by overtraining detector

    We ONLY want the exported voice models, never the G_/D_ checkpoints.
    """
    candidates: list[tuple[Path, int, bool]] = []  # (path, epoch, is_best)

    search_locations = [
        logs_dir,
        logs_dir / "weights",
        APPLIO_ROOT / "logs" / model_name,
        APPLIO_ROOT / "logs" / model_name / "weights",
        APPLIO_ROOT / "rvc" / "models",
    ]

    for loc in search_locations:
        if not loc.exists():
            continue
        for pth in loc.glob("*.pth"):
            name = pth.stem

            # Skip G_/D_ checkpoint files — these are training state, not voice models
            if name.startswith(("G_", "D_")):
                continue

            is_best = "best_epoch" in name
            epoch = 0
            try:
                for part in name.split("_"):
                    # Pattern: "300e" → epoch 300
                    if part.endswith("e") and part[:-1].isdigit():
                        epoch = int(part[:-1])
                    # Pattern: "e300" → epoch 300
                    elif part.startswith("e") and part[1:].isdigit():
                        epoch = int(part[1:])
                    # Pattern: "900s" → step 900 (use as tiebreaker)
                    elif part.endswith("s") and part[:-1].isdigit():
                        epoch = max(epoch, int(part[:-1]) // 3)  # rough step→epoch
            except (ValueError, IndexError):
                pass
            candidates.append((pth, epoch, is_best))

    if not candidates:
        return None

    # Prefer: best_epoch first, then highest epoch, then most recent
    candidates.sort(
        key=lambda x: (x[2], x[1], x[0].stat().st_mtime), reverse=True
    )
    best = candidates[0][0]
    log.info(f"Best model found: {best} ({best.stat().st_size / 1024 / 1024:.1f} MB)")
    return best


def _find_index(logs_dir: Path, model_name: str) -> Optional[Path]:
    """Find the FAISS .index file."""
    search_locations = [
        logs_dir,
        APPLIO_ROOT / "logs" / model_name,
    ]

    for loc in search_locations:
        if not loc.exists():
            continue
        for idx in loc.glob("*.index"):
            return idx

    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. CONVERT (inference) task
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _post_process_vocal(
    vocal_path: Path,
    output_path: Path,
    reverb_amount: float = 0.05,
    harmonic_enhance: bool = False,
    high_note_mode: bool = False,
    sample_rate: int = 44100,
    language: str = "auto",
) -> None:
    """Post-process converted vocal v49 — 한/영 분리 EQ + 동적 디에서 + 발음 보존.

    ── v49 (v45→v49 전면 개선) ──
    핵심 변경:
      - 한국어/영어 EQ 분리 (비음 포먼트 보호 vs 치찰음 대역 차이)
      - 3kHz presence boost 제거 (HiFi-GAN 이미 충분, 치찰음 증폭 주범)
      - 동적 디에서: adeclick → 고주파 아티팩트만 선택 제거
      - 300Hz/600Hz EQ: 한국어 비음 보호 위해 제거 (영어만 경미 감쇄)

    v49 체인 (한국어):
      highpass 70Hz → 8kHz -0.8dB/w=0.3 (HiFi-GAN 금속음만) →
      (리버브) → 2-pass loudnorm -14 LUFS (LRA=20)

    v49 체인 (영어):
      highpass 70Hz → 300Hz -0.3dB → 600Hz -0.3dB →
      5kHz -0.8dB/w=0.5 (s/sh 디에서) → 8kHz -0.8dB/w=0.3 →
      (리버브) → 2-pass loudnorm -14 LUFS (LRA=20)
    """
    filters = []

    # ━━━ 0. 노이즈 게이트 (RVC 추론 노이즈 제거) ━━━
    # v50: 무음 구간에 -65~-77dB RVC 노이즈 잔류 → 게이트로 제거
    # range_size=-50dB: 게이트 닫힐 때 -50dB 감쇄 (완전 무음 아닌 자연스러운 감쇠)
    # threshold=-45dB: -45dB 이하 신호 억제 (보컬은 보통 -30dB 이상)
    # attack/release: 부드러운 전환으로 클릭 방지
    filters.append("agate=threshold=0.006:range=0.003:attack=25:release=100")

    # ━━━ 1. 초저역 제거 (파열음 에너지 제어) ━━━
    filters.append("highpass=f=70:poles=2")

    # ━━━ 2. 언어별 EQ (v49: 한/영 분리) ━━━
    if language == "ko":
        # 한국어: 300Hz/600Hz EQ 없음 (비음 ㄴ/ㅁ/ㅇ 포먼트 보호)
        # v49.8: 1.2kHz 비음 컷은 아래 공통 Step 2b에서 적용
        pass
    elif language == "en":
        # 영어: 경미한 저역 감쇄만 (발음 명료도 유지)
        filters.append("equalizer=f=300:width_type=o:width=0.5:g=-0.3")
        filters.append("equalizer=f=600:width_type=o:width=0.7:g=-0.3")
        # 영어 치찰음은 4-6kHz에 집중 (s/sh/ch)
        filters.append("equalizer=f=5000:width_type=o:width=0.5:g=-0.8")
    else:
        # auto/기타: 영어 기본 + 비음 컷 (안전한 기본)
        filters.append("equalizer=f=300:width_type=o:width=0.5:g=-0.3")
        filters.append("equalizer=f=600:width_type=o:width=0.7:g=-0.3")
        filters.append("equalizer=f=5000:width_type=o:width=0.5:g=-0.8")

    # ━━━ 2b. HiFi-GAN 비음 공명 감쇄 (공통) ━━━
    # v49.8: 보코더가 800Hz-1.5kHz에서 비음 아티팩트 생성
    # v50: 1.2kHz -1.0→-0.5dB (고음 F1-F2 포먼트 보호, -1.0은 고음에서 얇은 소리)
    filters.append("equalizer=f=1200:width_type=o:width=0.4:g=-0.5")

    # ━━━ 3. HiFi-GAN 금속음 감쇄 (공통) ━━━
    # v49: 좁은 대역(0.3)으로 8kHz 금속성 아티팩트만 타겟팅
    # 넓은 대역 EQ는 공기감/숨소리도 함께 제거 → 좁게 제한
    filters.append("equalizer=f=8000:width_type=o:width=0.3:g=-0.8")

    # ━━━ v41: 후처리 리미터 제거 ━━━
    # v40: alimiter=limit=0.98:attack=5:release=100 → 이중 리미터(+mix limiter) = 펌핑
    # loudnorm + mix 리미터만으로 레벨 관리 충분

    # ━━━ 6. 리버브 (선택적) ━━━
    # v45: 8탭→4탭, decay 0.88→0.55 (더블링/코러스 효과 제거)
    # 커뮤니티 합의: RVC 변환에서 리버브는 최소화 또는 DAW에서 후처리 권장
    reverb_amount = max(0.0, min(0.5, float(reverb_amount)))
    if reverb_amount > 0.005:
        # 4탭 초기 반사음 — 소수 간격, 빠른 감쇠로 자연스러운 공간감만
        c1 = reverb_amount * 0.35
        c2 = reverb_amount * 0.18
        c3 = reverb_amount * 0.08
        c4 = reverb_amount * 0.03
        filters.append(
            f"aecho=1.0:0.55:"
            f"11|23|41|67:"
            f"{c1:.4f}|{c2:.4f}|{c3:.4f}|{c4:.4f}"
        )

    # EQ/리미터/리버브 처리
    _eq_tmp = output_path.with_suffix(".eq.wav")
    run_ffmpeg([
        "-i", str(vocal_path),
        "-af", ",".join(filters),
        "-acodec", "pcm_s24le",
        "-ar", str(sample_rate),
        str(_eq_tmp),
    ])

    # ━━━ 8. 2-pass Loudness 노멀라이즈 (EBU R128, -14 LUFS) ━━━
    # v36: 단일 패스 loudnorm linear=true는 측정 데이터 없이 비선형 폴백됨
    # 2-pass: Pass 1에서 측정 → Pass 2에서 선형 적용 → 다이나믹 레인지 완벽 보존
    try:
        import subprocess as _sp
        import json as _json
        # Pass 1: 측정
        _measure_cmd = [
            "ffmpeg", "-i", str(_eq_tmp), "-hide_banner",
            "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
            "-f", "null", "-"
        ]
        _measure = _sp.run(_measure_cmd, capture_output=True, text=True, timeout=120)
        # loudnorm JSON은 stderr 마지막에 출력됨
        _stderr = _measure.stderr
        _json_start = _stderr.rfind("{")
        _json_end = _stderr.rfind("}") + 1
        if _json_start >= 0 and _json_end > _json_start:
            _stats = _json.loads(_stderr[_json_start:_json_end])
            # Pass 2: 측정값으로 선형 노멀라이즈
            _ln_filter = (
                f"loudnorm=I=-14:TP=-1:LRA=11:linear=true"
                f":measured_I={_stats['input_i']}"
                f":measured_LRA={_stats['input_lra']}"
                f":measured_TP={_stats['input_tp']}"
                f":measured_thresh={_stats['input_thresh']}"
            )
            run_ffmpeg([
                "-i", str(_eq_tmp),
                "-af", _ln_filter,
                "-acodec", "pcm_s24le",
                "-ar", str(sample_rate),
                str(output_path),
            ])
            _eq_tmp.unlink(missing_ok=True)
            log.info(f"2-pass loudnorm applied: {_stats['input_i']} → -14 LUFS")
        else:
            # 측정 실패 시 EQ 결과를 그대로 사용
            _eq_tmp.rename(output_path)
            log.warning("loudnorm measurement failed, using EQ output as-is")
    except Exception as _ln_err:
        # loudnorm 전체 실패 시 EQ 결과를 그대로 사용
        if _eq_tmp.exists():
            if not output_path.exists():
                _eq_tmp.rename(output_path)
            else:
                _eq_tmp.unlink(missing_ok=True)
        log.warning(f"2-pass loudnorm failed: {_ln_err}, using EQ output")

    log.info(
        f"Vocal post-processed v43 → {output_path.name} "
        f"(reverb={reverb_amount:.2f}, high_note={high_note_mode}, "
        f"filters={len(filters)}, loudnorm=2pass)"
    )


def _mix_audio(
    vocal_path: Path,
    accomp_path: Path,
    output_path: Path,
    vocal_volume: float = 1.0,
    mr_volume: float = 1.0,
    sample_rate: int = 44100,
) -> None:
    """Mix converted vocals with original accompaniment (MR) v12.

    ── v12 믹싱 — 자동 RMS 레벨 매칭 + 단일 리미터 ──
    v11 핵심 문제:
      - 후처리 volume=1.75 + limiter + 믹싱 limiter = 이중 리미팅 → 펌핑/기계음
      - 고정 volume 승수 → 곡마다 다른 레벨에 대응 불가 → 과/부족 일관 발생

    v12 변경:
      - 후처리에서 볼륨 보상 완전 제거 → 이중 리미팅 해소
      - 자동 RMS 레벨 매칭: 보컬과 MR의 RMS 측정 → 보컬이 MR보다 +3.5dB 되도록 자동 조정
      - 곡별 자동 밸런스: 각 곡의 실제 레벨에 맞춰 최적 밸런스
      - MR 볼륨: *0.88 → *0.90 (반주 약간 더 보존)
      - MR EQ: 보컬 공간 확보 유지하되 최소한으로
      - 단일 리미터: 최종 출력에만 적용 (투명 리미팅)
    """
    import soundfile as _sf_mix

    # ── 자동 RMS 레벨 매칭 ──
    # 보컬과 MR의 RMS를 실측하여 적절한 밸런스 자동 계산
    # 타겟: 보컬 RMS = MR RMS × 1.5 (~+3.5dB, 팝/K-pop 보컬 프론트)
    try:
        _v_data, _v_sr = _sf_mix.read(str(vocal_path))
        _m_data, _m_sr = _sf_mix.read(str(accomp_path))
        if _v_data.ndim > 1:
            _v_data = _v_data.mean(axis=1)
        if _m_data.ndim > 1:
            _m_data = _m_data.mean(axis=1)

        _v_rms = float(np.sqrt(np.mean(_v_data ** 2)))
        _m_rms = float(np.sqrt(np.mean(_m_data ** 2)))

        if _v_rms > 1e-6 and _m_rms > 1e-6:
            _target_ratio = 1.5  # 보컬이 MR보다 +3.5dB 크게
            _auto_gain = (_m_rms * _target_ratio) / _v_rms
            _auto_gain = max(0.7, min(3.0, _auto_gain))  # v41: 0.5-6.0→0.7-3.0 (파열음 증폭 방지)
            log.info(
                f"Auto-balance: vocal_rms={20*np.log10(max(_v_rms,1e-10)):.1f}dBFS, "
                f"mr_rms={20*np.log10(max(_m_rms,1e-10)):.1f}dBFS, "
                f"auto_gain={_auto_gain:.2f} ({20*np.log10(_auto_gain):.1f}dB)"
            )
        else:
            _auto_gain = 1.5
            log.warning("RMS 측정 불가, 기본 auto_gain=1.5 사용")
    except Exception as _bal_err:
        _auto_gain = 1.5
        log.warning(f"Auto-balance 실패: {_bal_err}, 기본 gain=1.5 사용")

    _effective_vocal = vocal_volume * _auto_gain
    # 클리핑 방지: vocal gain이 4.0x(+12dB)를 넘지 않도록 제한
    if _effective_vocal > 4.0:
        log.warning(f"Vocal gain clamped: {_effective_vocal:.2f} → 4.0 (clipping prevention)")
        _effective_vocal = 4.0

    run_ffmpeg([
        "-i", str(vocal_path),
        "-i", str(accomp_path),
        "-filter_complex",
        # ── 보컬 체인 ──
        # v12: 후처리에서 볼륨 보상 제거 → 여기서 자동 레벨 매칭 적용
        # v17: stereotools 제거 — slev=0.05는 사실상 무음(+0.04dB), 불필요한 필터 레이턴시만 발생
        f"[0:a]aresample=resampler=soxr,volume={_effective_vocal:.3f},"
        f"aformat=channel_layouts=stereo[v];"
        # ── MR 체인 ──
        # v45: 보컬 블리드 감쇄 강화 (Demucs 잔류 보컬이 더블링 유발)
        # 800Hz~4kHz 대역은 보컬 포먼트 핵심 → MR에서 강하게 감쇄
        f"[1:a]aresample=resampler=soxr,volume={mr_volume * 0.90:.3f},"
        f"lowshelf=f=60:width_type=o:width=0.8:g=-0.8,"
        f"equalizer=f=800:width_type=o:width=1.5:g=-1.5,"   # v45: -1.0→-1.5 (보컬 블리드)
        f"equalizer=f=2000:width_type=o:width=1.0:g=-2.0,"  # v45: 추가 (2kHz 보컬 핵심대역)
        f"equalizer=f=3500:width_type=o:width=0.8:g=-1.5[m];"  # v45: 2.5k→3.5k, -1.0→-1.5
        # ── 최종 믹스 + 단일 리미터 ──
        f"[v][m]amix=inputs=2:duration=longest:normalize=0,"
        f"alimiter=limit=0.89:attack=25:release=300:level=disabled",  # v45: level=disabled (자동 게인 올림 방지 → 클리핑 제거)
        "-acodec", "pcm_s24le",
        "-ar", str(sample_rate),
        str(output_path),
    ])
    # 출력 파일 검증
    if not output_path.exists():
        raise RuntimeError(f"믹싱 출력 파일이 생성되지 않았습니다: {output_path}")
    out_size = output_path.stat().st_size
    if out_size < 10_000:
        raise RuntimeError(f"믹싱 출력 파일 크기 이상 ({out_size}바이트): {output_path}")
    log.info(f"Mixed audio v12: {output_path.name} (vocal_vol={vocal_volume}, "
             f"auto_gain={_auto_gain:.2f}, mr_vol={mr_volume}, "
             f"size={out_size / 1024 / 1024:.1f}MB)")


def _pre_filter_vocal_harmony(
    vocal_path: Path,
    output_path: Path,
    strength: float = 0.5,
) -> bool:
    """v22: Pre-RVC 보컬 화음 억제 — HPSS 기반 (⚠️ v23에서 기본 비활성).

    ⚠️ v23 분석 결과: HPSS는 harmonic/percussive 분리이므로 보컬의 자음/숨소리/
    발음(percussive 성분)까지 제거하여 웅얼웅얼 발음 + 기계음 심화 확인됨.
    - ZCR -17%, Presence -2dB, Air -2.3dB 손실 → 발음 인식 불가 수준
    - 모든 프리셋에서 harmonyFilter=0 으로 비활성화됨
    - 향후 대안: 리드 보컬 분리 모델 (Demucs lead vocal stem) 검토 필요

    strength:
      0.0 = 비활성 (bypass) ← v23 기본값
    """
    try:
        import soundfile as _sf_hf
        import numpy as _np_hf
        import librosa as _lib_hf

        if strength <= 0.01:
            return False

        y, sr = _sf_hf.read(str(vocal_path))
        is_stereo = y.ndim > 1
        y_mono = y.mean(axis=1) if is_stereo else y.copy()

        hop = 512
        n_fft = 2048
        S = _lib_hf.stft(y_mono.astype(float), n_fft=n_fft, hop_length=hop)

        # HPSS: margin은 strength에 비례 (1.5~4.0)
        # margin 높을수록 harmonic/percussive 분리가 공격적
        margin = 1.5 + strength * 2.5  # 0.3→2.25, 0.5→2.75, 0.8→3.5, 1.0→4.0
        H, P = _lib_hf.decompose.hpss(S, margin=margin)

        # HPSS harmonic만 사용 → 배경 화음의 percussive/noise 성분 제거
        y_filtered = _lib_hf.istft(H, hop_length=hop, length=len(y_mono))

        # 원본 RMS에 맞게 레벨 매칭 (HPSS로 에너지 손실 보상)
        rms_orig = _np_hf.sqrt(_np_hf.mean(y_mono ** 2))
        rms_filt = _np_hf.sqrt(_np_hf.mean(y_filtered ** 2))
        if rms_filt > 1e-10:
            y_filtered *= rms_orig / rms_filt

        # 스테레오 원본이면 mono 결과를 양 채널에 복사
        if is_stereo:
            y_out = _np_hf.column_stack([y_filtered, y_filtered])
        else:
            y_out = y_filtered

        _sf_hf.write(str(output_path), y_out, sr)
        log.info(
            f"Pre-RVC harmony filter v22: strength={strength:.2f} margin={margin:.1f} "
            f"→ {output_path.name}"
        )
        return True

    except Exception as _hf_err:
        log.warning(f"Pre-RVC harmony filter failed (non-critical, skipping): {_hf_err}")
        return False


def _fix_pitch_artifacts(
    vocal_path: Path,
    output_path: Path,
    max_hz: float = 1200.0,
    min_duration_s: float = 0.20,
    gap_bridge_s: float = 0.15,
    vp_threshold: float = 0.005,  # v50: 0.01→0.005 (가성 오탐 방지 — 0.01은 진짜 가성도 억제)
) -> bool:
    """RVC 변환 후 고음역 아티팩트 감지·감쇠 (v19).

    Demucs가 악기 신호(피아노/신스 고음역)를 보컬 트랙에 포함시킬 때
    RVC가 해당 신호를 괴성으로 변환하는 문제 수정.

    v14 문제: min_duration_s=0.8s → 실측 아티팩트가 0.02-0.30s 짧은 버스트로
    분산되어 모두 누락. (105s 0.30s, 106.9s 0.12s, 107.1s 0.23s 등)

    v15 수정:
      - min_duration_s: 0.8 → 0.08 (80ms 이상 버스트 감지)
      - gap_bridge_s: 0.30s 이하 갭은 같은 아티팩트 이벤트로 연결
      - max_hz: 480 → 490 (B4=494Hz 경계 위 명확한 아티팩트만 대상)

    v16 수정:
      - Zone 2 VP 기반 감지 추가: 430-490Hz 구간에서 PYIN 신뢰도(voiced_prob)
        < vp_threshold(0.12)인 프레임도 아티팩트로 처리
      - 실측: 기다릴게12 103-115s 괴성 구간 430-490Hz 아티팩트 VP=0.010~0.092
        (490Hz 임계값 이하라 기존 Zone1에서 누락되던 구간 보완)
      - 실제 가창 프레임: VP ≥ 0.20 (지속구간) → Zone2에 해당 없음
      - vp_threshold 파라미터로 민감도 조절 가능 (기본값 0.12)

    v18 수정:
      - max_hz: 490 → 530 (C5=523Hz: 여성보컬/남성가성 상한 보호, 여성보컬 F0 최대 504Hz)
      - vp_threshold: 0.12 → 0.06 (실측: 실제 가성 VP=0.06~0.12, 악기 누화 VP<0.05)
        기존 0.12 → 기다릴게 35-38s, 95-98s, 108-110s 가성 묵음 역효과 수정
      - min_duration_s: 0.08 → 0.20 (200ms 미만 단발 버스트 보존: 짧은 가성 음절 보호)
      - gap_bridge_s: 0.30 → 0.15 (과도한 버스트 연결 방지: 화음 영역 실제 가창 훼손 방지)

    v19 수정 (기다릴게14 HPSS 실측 분석 기반):
      - max_hz: 530 → 1200 (Zone 1 사실상 비활성화)
        실측: 기다릴게 가성 F0=519-809Hz → max_hz=530이 C5+(523Hz) 이상 가성 전체 게이팅
        35-38s: 519-543Hz, 95-98s: 480-540Hz, 108-110s: 530-613Hz, 115-118s: 596-809Hz
        htdemucs_6s 전환으로 피아노/기타 누화 감소 → Zone 1의 존재 이유 희박
        1200Hz 이상은 인간 음성으로 나올 수 없으므로 실질적 아티팩트만 게이팅
      - vp_threshold: 0.06 → 0.03 (더 보수적 Zone 2)
        실측: 변환 보컬 가성 구간 VP가 원곡보다 낮을 수 있어 0.06이 실제 가성도 게이팅
        vp_threshold=0.03: 확실한 악기 누화 아티팩트(VP=0.01~0.02)만 게이팅
      - Zone 2 범위 고정: 430-530Hz (max_hz와 독립)
        max_hz=1200으로 올려도 Zone 2가 430-1200Hz로 확대되는 것 방지
        530Hz+ 가성 영역은 Zone 2에서 제외 (실제 가성 보호)

    v20 수정 → v21에서 되돌림:
      - Zone 3 삭제: 80Hz/4회 파라미터가 원곡에서도 38.1% 오탐 발생
        v16에서 voiced 프레임의 57.7%를 -10dB 억제 → 전체 곡 품질 파괴
        화음 구간 괴성은 RVC 아키텍처 한계 — 후처리 게이팅으로 해결 불가
      - EQ v19 복원: v20 중립화(+0.5dB/없음)가 발음선명도·고역디테일 저하
    """
    try:
        import soundfile as _sf_pa
        import numpy as _np_pa
        import librosa as _lib_pa

        y, sr = _sf_pa.read(str(vocal_path))
        is_stereo = y.ndim > 1
        y_mono = y.mean(axis=1) if is_stereo else y.copy()

        hop = 512
        f0, voiced, voiced_prob = _lib_pa.pyin(
            y_mono, fmin=80, fmax=1200, sr=sr, frame_length=2048, hop_length=hop
        )
        # Zone 1: F0 > max_hz (v19: 1200Hz — 인간 목소리 최상한 이상의 극한 아티팩트만)
        # htdemucs_6s로 피아노/기타 분리 후 Zone 1 필요성 감소
        zone1 = voiced & ~_np_pa.isnan(f0) & (f0 > max_hz)
        # Zone 2: 430-530Hz 고정 범위, PYIN 신뢰도 < vp_threshold (악기 아티팩트 시그니처)
        # v19: 범위를 430-530Hz로 고정 (max_hz와 독립) — 530Hz+ 가성 영역 보호
        # 실측: 기다릴게 가성 F0=519-809Hz, VP=0.06~0.83 → vp_threshold=0.03으로 실제 가성 통과
        # 악기 누화 특성: VP=0.01~0.02 (확실한 아티팩트만 게이팅)
        _zone2_max_hz = 530.0  # max_hz와 독립적인 Zone 2 상한 (고정)
        zone2 = (
            voiced & ~_np_pa.isnan(f0)
            & (f0 > 430.0) & (f0 <= _zone2_max_hz)
            & (voiced_prob < vp_threshold)
        )
        artifact = zone1 | zone2

        # v21: Zone 3 제거 — 80Hz/4회 파라미터가 원곡에서 38.1% 오탐
        # v16에서 voiced 57.7%를 -10dB 억제하여 전체 곡 파괴
        # 화음 괴성은 RVC 다성부 입력 한계 — 후처리 게이팅으로 해결 불가

        # --- gap bridging: gap_bridge_s 이하 간격의 아티팩트 버스트를 연결 ---
        gap_frames = max(1, int(gap_bridge_s * sr / hop))
        bridged = artifact.copy()
        i = 0
        while i < len(bridged):
            if bridged[i]:
                # 현재 버스트 끝 찾기
                j = i
                while j < len(bridged) and bridged[j]:
                    j += 1
                # 다음 버스트 탐색 (gap_frames 이내)
                k = j
                while k < min(j + gap_frames, len(bridged)) and not bridged[k]:
                    k += 1
                if k < len(bridged) and bridged[k] and (k - j) <= gap_frames:
                    bridged[j:k] = True  # 갭 채우기
                    # 연결 후 현재 위치 유지 (확장된 버스트 재탐색)
                    continue
                i = j
            else:
                i += 1
        artifact = bridged

        # v21: Zone3 제거됨 — Zone1/2만 억제 (v19 동작 복원)
        min_frames = max(1, int(min_duration_s * sr / hop))
        gain = _np_pa.ones(len(f0), dtype=float)
        suppressed_count = 0
        i = 0
        while i < len(artifact):
            if artifact[i]:
                j = i
                while j < len(artifact) and artifact[j]:
                    j += 1
                if (j - i) >= min_frames:
                    fade = max(1, int(0.1 * sr / hop))
                    for k in range(i, j):
                        if k < i + fade:
                            gain[k] = max(0.30, 1.0 - (k - i) / fade * 0.70)
                        elif k >= j - fade:
                            gain[k] = max(0.30, 1.0 - (j - k) / fade * 0.70)
                        else:
                            gain[k] = 0.30  # v50: 0.15→0.30 (≈ -10 dB, 가성 오탐 시 덜 공격적)
                    _med_f0 = float(_np_pa.nanmedian(f0[i:j]))
                    _zone = "Z1" if _med_f0 > max_hz else f"Z2(VP<{vp_threshold})"
                    log.info(
                        f"Pitch artifact gate v21 [{_zone}]: {i*hop/sr:.2f}s–{j*hop/sr:.2f}s "
                        f"F0≈{_med_f0:.0f}Hz suppressed"
                    )
                    suppressed_count += 1
                i = j
            else:
                i += 1

        if suppressed_count == 0:
            return False

        gain_audio = _np_pa.interp(
            _np_pa.arange(len(y_mono)),
            _np_pa.arange(len(gain)) * hop + hop // 2,
            gain,
        )
        gain_audio = _np_pa.clip(gain_audio, 0.0, 1.0)
        y_fixed = (y * gain_audio[:, _np_pa.newaxis]) if is_stereo else (y * gain_audio)
        _sf_pa.write(str(output_path), y_fixed, sr)
        log.info(f"Pitch artifact gate v21: {suppressed_count} segment(s) suppressed → {output_path.name}")
        return True

    except Exception as _pa_err:
        log.warning(f"Pitch artifact gate failed (non-critical, skipping): {_pa_err}")
        return False


def task_convert(job_input: dict, job: dict) -> dict:
    """
    RVC v2 voice conversion (SVC pipeline):
      1) Decode model (.pth, .index) and input audio
      2) Demucs vocal separation → vocals + accompaniment (MR)
      2c) [선택] vocal_pitch_pre_shift — 여성/고음 파트 사전 피치 조정
      3) RVC voice conversion on vocals only
      3a-fix2) 피치 아티팩트 게이트 — A5+ 지속 구간 감쇠
      3b) Post-process vocals: EQ, compression, optional harmonic enhance + room reverb
      3c) [선택] vocal_pitch_post_shift — 사전 조정 복원
      4) Mix post-processed vocals + original MR
      5) Return converted vocals + mixed output as base64
    """
    pth_b64: str = job_input.get("pth_data", "")
    pth_url: str = job_input.get("pth_url", "")
    index_b64: str = job_input.get("index_data", "")
    index_url: str = job_input.get("index_url", "")
    audio_b64: str = job_input.get("audio_data", "")
    audio_url: str = job_input.get("audio_url", "")
    audio_filename: str = Path(job_input.get("audio_filename", "input.wav")).name  # path traversal 방지
    # 명시적 타입 변환: RunPod job_input에서 문자열로 전달될 수 있음
    # try-except: 잘못된 값이 들어와도 기본값으로 폴백 (task_train과 동일 패턴)
    try:
        pitch_shift: int = int(job_input.get("pitch_shift", 0))
    except (ValueError, TypeError):
        pitch_shift = 0
    # index_rate 0.45: v51 (v50: 0.55→0.45, 발음 명료도 우선 — 0.55는 발음 뭉개짐)
    try:
        index_rate: float = float(job_input.get("index_rate", 0.45))
    except (ValueError, TypeError):
        index_rate = 0.45
    # rmvpe: stable, fast, accurate for singing — better default than crepe
    _VALID_F0_CONVERT = {"rmvpe", "fcpe", "crepe", "crepe-tiny", "harvest", "pm"}
    f0_method: str = job_input.get("f0_method", "rmvpe")
    if f0_method not in _VALID_F0_CONVERT:
        log.warning(f"Invalid f0_method '{f0_method}', falling back to rmvpe")
        f0_method = "rmvpe"
    # filter_radius 2: v50 (v49: 3→v50: 2, 고음 비브라토 보존 — 3은 61ms 스무딩으로 가성 평탄화)
    try:
        filter_radius: int = int(job_input.get("filter_radius", 2))  # v50: 3→2
    except (ValueError, TypeError):
        filter_radius = 2
    # rms_mix_rate 0.1: v51 — 원곡 음량 패턴 10% 반영 (음량 균일성 개선)
    # v36: 0.0 (100% 모델 다이나믹) → 음량 들쑥날쑥 문제 발생
    # v51: 0.1로 원곡 엔벨로프 약간 반영 → 원곡과 유사한 음량 패턴
    try:
        rms_mix_rate: float = float(job_input.get("rms_mix_rate", 0.1))
    except (ValueError, TypeError):
        rms_mix_rate = 0.1
    # protect 0.40: v49 (0.33→0.40, 과도한 자음보호 완화→인덱스 정확도↑)
    # RVC 구현: 0.50은 보호 기능 OFF, 0.0은 최대 보호
    try:
        protect: float = float(job_input.get("protect", 0.40))
    except (ValueError, TypeError):
        protect = 0.40
    # v49: hop_length 128 (커뮤니티 표준, 64는 노이즈 추적→삑사리)
    try:
        hop_length: int = int(job_input.get("hop_length", 128))
    except (ValueError, TypeError):
        hop_length = 128
    # v49: 한국어/영어 EQ 분리를 위한 language 파라미터
    language: str = str(job_input.get("language", "auto")).lower().strip()
    if language not in ("ko", "en", "auto"):
        language = "auto"
    # v49.7: f0_autotune을 job_input에서 받음 (하드코딩 제거)
    _autotune_raw = job_input.get("f0_autotune", True)
    f0_autotune: bool = _autotune_raw in (True, "true", "True", "1", 1)
    try:
        f0_autotune_strength: float = float(job_input.get("f0_autotune_strength", 0.5))
    except (ValueError, TypeError):
        f0_autotune_strength = 0.5  # v50: 0.3→0.5 (음정 단단하게, 0.3은 너무 약함)
    f0_autotune_strength = max(0.0, min(1.0, f0_autotune_strength))
    clean_audio_raw = job_input.get("clean_audio", False)
    clean_audio: bool = clean_audio_raw in (True, "true", "True", "1", 1)
    try:
        clean_strength: float = float(job_input.get("clean_strength", 0.7))
    except (ValueError, TypeError):
        clean_strength = 0.7
    export_format: str = job_input.get("export_format", "wav").lower().strip()
    # SVC pipeline options — 일관된 in() 패턴으로 boolean 파싱
    separate_vocals_raw = job_input.get("separate_vocals", True)
    separate_vocals: bool = separate_vocals_raw in (True, "true", "True", "1", 1)
    try:
        vocal_volume: float = float(job_input.get("vocal_volume", 1.0))
    except (ValueError, TypeError):
        vocal_volume = 1.0
    try:
        mr_volume: float = float(job_input.get("mr_volume", 1.0))
    except (ValueError, TypeError):
        mr_volume = 1.0
    # Post-processing for naturalness (applied after RVC conversion)
    try:
        post_reverb: float = float(job_input.get("post_reverb", 0.0))
    except (ValueError, TypeError):
        post_reverb = 0.0
    harmonic_enhance_raw = job_input.get("harmonic_enhance", False)
    harmonic_enhance: bool = harmonic_enhance_raw in (True, "true", "True", "1", 1)
    # 고음/가성 최적화 모드: 후처리 EQ를 가성 친화적으로 조정
    high_note_mode_raw = job_input.get("high_note_mode", False)
    high_note_mode: bool = high_note_mode_raw in (True, "true", "True", "1", 1)
    # v22: Pre-RVC 화음 필터 (HPSS) — v23에서 기본 비활성
    # ⚠️ HPSS는 자음/숨소리(percussive)까지 제거하여 발음 파괴 확인됨
    # 0.0=비활성(기본), UI에서 수동 활성화 가능하나 비권장
    try:
        harmony_filter: float = float(job_input.get("harmony_filter", 0.0))
    except (ValueError, TypeError):
        harmony_filter = 0.0

    # 파라미터 범위 클램프
    pitch_shift = max(-24, min(24, pitch_shift))
    index_rate = max(0.0, min(1.0, index_rate))
    filter_radius = max(1, min(32, filter_radius))  # min 1: 0은 일부 RVC 버전에서 오류
    rms_mix_rate = max(0.0, min(1.0, rms_mix_rate))
    protect = max(0.0, min(1.0, protect))
    clean_strength = max(0.0, min(1.0, clean_strength))
    post_reverb = max(0.0, min(0.5, post_reverb))
    hop_length = max(1, min(512, hop_length))
    vocal_volume = max(0.0, min(2.0, vocal_volume))
    mr_volume = max(0.0, min(2.0, mr_volume))
    harmony_filter = max(0.0, min(1.0, harmony_filter))
    # v36: 원본 보컬 블렌딩 비율 (숨결감/자연스러움 복원)
    # 0.0=비활성, 0.10~0.20=권장 (원본 보컬의 숨소리/자음 10~20% 혼합)
    try:
        vocal_blend: float = float(job_input.get("vocal_blend", 0.0))
    except (ValueError, TypeError):
        vocal_blend = 0.0
    vocal_blend = max(0.0, min(0.3, vocal_blend))

    if not pth_b64 and not pth_url:
        raise ValueError("No model provided (pth_data or pth_url required)")
    if not audio_b64 and not audio_url:
        raise ValueError("No audio provided (audio_data or audio_url required)")
    # Whitelist export_format to prevent path traversal / unexpected filenames
    _VALID_FORMATS = {"wav", "mp3", "flac", "ogg", "m4a"}
    if export_format not in _VALID_FORMATS:
        log.warning(f"Invalid export_format '{export_format}', defaulting to 'wav'")
        export_format = "wav"

    job_id = job.get("id") or uuid.uuid4().hex[:12]
    work = ensure_dir(WORK_DIR / f"convert_{job_id}")
    start_time = time.time()

    try:
        # --- Step 1: Decode files ---
        runpod.serverless.progress_update(job, "모델·오디오 파일 다운로드 중... (1/4)")

        # Model: download from URL or decode base64
        if pth_url:
            import requests as _req
            log.info(f"Downloading model from URL: {pth_url[:80]}...")
            pth_path = work / "model.pth"
            _download_with_retries(_req, pth_url, pth_path, "모델 파일", timeout=120)
        else:
            pth_path = decode_b64_file(pth_b64, work / "model.pth")
        # 모델 파일 기본 검증 (크기 + 파일 형식)
        pth_size = pth_path.stat().st_size
        if pth_size < 1_000_000:  # 1MB 미만은 비정상
            raise RuntimeError(f"모델 파일이 너무 작습니다 ({pth_size / 1024:.1f} KB). 손상되었거나 잘못된 파일입니다.")
        log.info(f"Model file ready: {pth_size / 1024:.1f} KB")

        # Index: download from URL or decode base64
        index_path = None
        if index_url:
            import requests as _req
            log.info(f"Downloading index from URL: {index_url[:80]}...")
            try:
                _download_with_retries(_req, index_url, work / "model.index", "인덱스 파일", timeout=60)
                index_path = work / "model.index"
            except Exception as dl_err:
                log.warning(f"인덱스 파일 다운로드 실패 (선택사항, 계속 진행): {dl_err}")
                index_path = None
        elif index_b64:
            index_path = decode_b64_file(index_b64, work / "model.index")
        if index_path:
            log.info(f"Index file ready: {index_path.stat().st_size / 1024:.1f} KB")
        else:
            # 인덱스 없으면 index_rate 무의미 → 0으로 리셋 (FAISS 조회 방지)
            if index_rate > 0:
                log.info(f"No index file available, resetting index_rate {index_rate} → 0")
                index_rate = 0.0

        # Decode input audio: download from URL or decode base64
        input_ext = Path(audio_filename).suffix.lower() or ".mp3"
        if audio_url:
            import requests as _req
            log.info(f"Downloading audio from URL: {audio_url[:80]}...")
            raw_input = work / f"input{input_ext}"
            _download_with_retries(_req, audio_url, raw_input, "오디오 파일", timeout=120)
        else:
            raw_input = decode_b64_file(audio_b64, work / f"input{input_ext}")
        log.info(f"Audio file ready: {raw_input.stat().st_size / 1024:.1f} KB")

        # v36: 원본 샘플레이트 감지 및 보존
        # 48kHz WAV 등 고음질 소스의 다운샘플 방지
        import soundfile as _sf_sr
        try:
            _src_info = _sf_sr.info(str(raw_input))
            _orig_sr = _src_info.samplerate
        except Exception:
            _orig_sr = 44100
        # RVC는 내부적으로 16kHz로 리샘플하지만, 최종 출력은 원본 SR 보존
        # 중간 처리는 44.1kHz 또는 48kHz 중 원본에 가까운 값 사용
        _process_sr = 48000 if _orig_sr >= 48000 else 44100
        log.info(f"Original sample rate: {_orig_sr}Hz → processing at {_process_sr}Hz")

        # Normalize to WAV for processing (keep STEREO for Demucs quality)
        # soxr precision=28: 최고품질 리샘플링 — 원음 주파수 특성 최대 보존
        # pcm_s24le: 24-bit PCM (16-bit 대비 양자화 노이즈 48dB 감소)
        input_stereo = work / "input_stereo.wav"
        run_ffmpeg([
            "-i", str(raw_input),
            "-af", f"aresample=resampler=soxr:precision=28:osr={_process_sr}",
            "-acodec", "pcm_s24le",
            "-ar", str(_process_sr),
            str(input_stereo),
        ])
        # Also create mono version for direct RVC use (when skipping Demucs)
        input_wav = work / "input_normalized.wav"
        run_ffmpeg([
            "-i", str(raw_input),
            "-af", f"aresample=resampler=soxr:precision=28:osr={_process_sr}",
            "-acodec", "pcm_s24le",
            "-ar", str(_process_sr),
            "-ac", "1",
            str(input_wav),
        ])

        # --- Step 2: Vocal separation (BS-Roformer → Demucs fallback) ---
        accomp_path = None
        if separate_vocals:
            # v36: BS-Roformer 우선 시도 (SDR 12.9, SOTA)
            # 실패 시 Demucs로 폴백 (SDR ~8.5, 안정적)
            roformer_result = None
            try:
                runpod.serverless.progress_update(job, "Separating vocals (BS-Roformer)... (2/4)")
                roformer_dir = ensure_dir(work / "roformer")
                roformer_result = _roformer_separate(input_stereo, roformer_dir)
            except Exception as e:
                log.warning(f"BS-Roformer attempt failed: {e}")

            if roformer_result and roformer_result.get("vocals"):
                rvc_input = roformer_result["vocals"]
                accomp_path = roformer_result.get("accompaniment")
                log.info("Using BS-Roformer separation (SOTA quality)")
            else:
                # Demucs fallback
                runpod.serverless.progress_update(job, "Separating vocals (Demucs)... (2/4)")
                demucs_dir = ensure_dir(work / "demucs")
                separation = _demucs_separate([input_stereo], demucs_dir)

                if separation["vocals"]:
                    rvc_input = separation["vocals"][0]
                else:
                    log.warning("Demucs produced no vocals, using original audio for RVC")
                    rvc_input = input_wav

                if separation["accompaniment"]:
                    accomp_path = separation["accompaniment"][0]
                    log.info(f"Accompaniment saved: {accomp_path.name}")
        else:
            rvc_input = input_wav

        # --- Step 2b: Pre-RVC 화음 필터 (v22, v23 기본 비활성) ---
        # ⚠️ HPSS는 자음/발음(percussive)을 파괴 → v23에서 모든 프리셋 0.0
        # 사용자가 UI에서 수동 활성화한 경우에만 실행
        if harmony_filter > 0.01 and rvc_input.exists():
            log.warning(f"⚠️ HPSS harmony filter activated (strength={harmony_filter:.2f})"
                        " — v23에서 비권장: 자음/발음 파괴 위험")
            _hf_out = work / "vocal_harmony_filtered.wav"
            if _pre_filter_vocal_harmony(rvc_input, _hf_out, strength=harmony_filter):
                rvc_input = _hf_out
                log.info(f"Pre-RVC harmony filter applied: strength={harmony_filter:.2f}")
            else:
                log.info("Pre-RVC harmony filter skipped (no change)")

        # --- Step 2c: 리드/백킹 보컬 분리 (v49.5 — 화음 처리) ---
        # 문제: Demucs/BS-Roformer는 모든 보컬(리드+화음)을 1개 스템으로 분리
        # → RVC가 전부 동일 음색으로 변환 → 화음이 부자연스럽고 이상한 소리
        # 해결: 리드 보컬만 RVC 변환, 백킹/화음은 원본 유지
        backing_vocals_path = None
        if separate_vocals and rvc_input.exists():
            try:
                runpod.serverless.progress_update(job, "Separating lead/backing vocals... (2.5/4)")
                lead_back_dir = ensure_dir(work / "lead_backing")
                lb_result = _separate_lead_backing(rvc_input, lead_back_dir)
                if lb_result.get("lead") and lb_result["lead"].exists():
                    backing_vocals_path = lb_result.get("backing")
                    rvc_input = lb_result["lead"]  # 리드만 RVC 변환
                    log.info(f"Lead/backing split: lead→RVC, backing→original "
                             f"(backing={backing_vocals_path.name if backing_vocals_path else 'N/A'})")
                else:
                    log.info("Lead/backing separation unavailable, converting full vocal stem")
            except Exception as lb_err:
                log.warning(f"Lead/backing separation failed: {lb_err}, continuing with full vocals")

        # --- Step 2b-old: Pre-RVC 디에서 제거됨 ---
        # 이전: 7kHz Q=3 -2dB + 후처리 8.5kHz -1dB = 이중 디에싱 → 한국어 자음 포먼트(4-8kHz) 파괴
        # 한국어 마찰음(ㅅ,ㅆ,ㅈ,ㅊ,ㅎ)의 에너지가 4-8kHz에 집중 → 이 대역 커팅은 발음 뭉개짐 유발
        # → 디에싱은 후처리에서 한 번만 적용 (9kHz 이상의 좁은 대역만 타겟)

        # --- Vocal pitch pre-shift: 제거됨 (v15) ---
        # librosa.effects.pitch_shift STFT phase vocoder는 formant 미보존 → 기계음 유발.
        # 향후 rubberband 등 formant-preserving 피치 시프터 도입 시 재구현 예정.

        # --- Step 3: RVC voice conversion on vocals ---
        runpod.serverless.progress_update(job, "Running voice conversion (RVC)... (3/4)")

        # 오디오 길이 검증 (너무 짧으면 RVC가 빈 출력이나 오류 반환)
        import soundfile as _sf_check
        try:
            _rvc_info = _sf_check.info(str(rvc_input))
            _rvc_duration = _rvc_info.frames / _rvc_info.samplerate
        except Exception as _dur_err:
            log.warning(f"Audio duration check failed: {_dur_err}, defaulting to 60s")
            _rvc_duration = 60.0
        if _rvc_duration < 0.3:
            raise RuntimeError(
                f"입력 오디오가 너무 짧습니다 ({_rvc_duration:.2f}초). "
                "최소 0.3초 이상의 오디오가 필요합니다."
            )
        if _rvc_duration > 300:
            log.warning(f"긴 오디오 입력 ({_rvc_duration:.0f}초). 변환에 시간이 오래 걸릴 수 있습니다.")
        # v49: 3분 이하는 분할 없이 처리 → 청크 경계 아티팩트(가사 끊김/음 끊김) 방지
        # 3분 이상은 GPU OOM 방지를 위해 분할 처리 (5분→3분: 긴 곡 청크 경계 안정성)
        _should_split = _rvc_duration > 180

        converted_vocals_path = work / f"converted_vocals.{export_format}"
        _rvc_infer(
            pth_path=pth_path,
            index_path=index_path,
            input_audio=rvc_input,
            output_path=converted_vocals_path,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            hop_length=hop_length,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            split_audio=_should_split,
            f0_autotune=f0_autotune,              # v49.7: job_input에서 받음 (기본 True)
            f0_autotune_strength=f0_autotune_strength,  # v49.7: job_input에서 받음 (기본 0.6)
        )

        if not converted_vocals_path.exists():
            # 작업 디렉토리 내 파일 목록 로깅
            all_files = list(work.rglob("*"))
            log.error(f"Expected output: {converted_vocals_path}")
            log.error(f"All files in work dir: {[str(f.relative_to(work)) for f in all_files if f.is_file()]}")
            raise RuntimeError(
                f"Conversion produced no output file at {converted_vocals_path.name}. "
                f"Work dir contains: {[f.name for f in all_files if f.is_file()]}"
            )

        # --- Step 3a-fix: RVC 출력 SR 정규화 (40kHz→44.1kHz) ---
        # RVC 모델은 보통 tgt_sr=40000Hz로 출력, 후처리/믹싱은 44.1kHz 기준.
        # SR 불일치 시 soxr 최고품질 리샘플링으로 44.1kHz 통일.
        import soundfile as _sf_sr
        try:
            _rvc_out_info = _sf_sr.info(str(converted_vocals_path))
            _rvc_out_sr = _rvc_out_info.samplerate
            if _rvc_out_sr != _process_sr:
                log.info(f"RVC output SR={_rvc_out_sr} → resampling to {_process_sr}Hz (soxr)")
                _resampled_path = work / f"converted_vocals_{_process_sr // 1000}k.wav"
                run_ffmpeg([
                    "-i", str(converted_vocals_path),
                    "-af", f"aresample=resampler=soxr:precision=28:osr={_process_sr}",
                    "-acodec", "pcm_s24le",
                    "-ar", str(_process_sr),
                    str(_resampled_path),
                ])
                if _resampled_path.exists() and _resampled_path.stat().st_size > 1000:
                    converted_vocals_path = _resampled_path
        except Exception as _sr_err:
            log.warning(f"SR check/resample failed, continuing with original: {_sr_err}")

        # --- Step 3a-verify: 출력 길이 검증 (가사 끊김/끊어짐 감지) ---
        try:
            _out_dur = get_audio_duration(converted_vocals_path)
            _dur_ratio = _out_dur / max(_rvc_duration, 0.1)
            if _dur_ratio < 0.90:
                # 이전 임계값 0.85 → 0.90: 10% 이상 잘리면 재시도
                log.error(
                    f"⚠️ OUTPUT TRUNCATED: input={_rvc_duration:.1f}s, output={_out_dur:.1f}s "
                    f"(ratio={_dur_ratio:.2f}). 가사가 잘릴 수 있습니다."
                )
                # split_audio=False였다면 True로 재시도 (30초 이상이면 시도)
                if not _should_split and _rvc_duration > 30:
                    log.info("Retrying with split_audio=True to prevent truncation...")
                    _retry_path = work / f"converted_vocals_retry.{export_format}"
                    _rvc_infer(
                        pth_path=pth_path,
                        index_path=index_path,
                        input_audio=rvc_input,
                        output_path=_retry_path,
                        pitch_shift=pitch_shift,
                        index_rate=index_rate, f0_method=f0_method,
                        filter_radius=filter_radius, rms_mix_rate=rms_mix_rate,
                        protect=protect, hop_length=hop_length,
                        clean_audio=clean_audio, clean_strength=clean_strength,
                        export_format=export_format, split_audio=True,
                        f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength,
                    )
                    if _retry_path.exists():
                        _retry_dur = get_audio_duration(_retry_path)
                        if _retry_dur > _out_dur:
                            log.info(f"Retry improved: {_out_dur:.1f}s → {_retry_dur:.1f}s")
                            converted_vocals_path = _retry_path
                            _out_dur = _retry_dur
                            _dur_ratio = _out_dur / max(_rvc_duration, 0.1)

                # 재시도 후에도 여전히 잘린 경우 — 끝부분 무음 패딩 추가
                if _dur_ratio < 0.95:
                    _missing_sec = _rvc_duration - _out_dur
                    if 0 < _missing_sec < 15:
                        log.info(f"Adding {_missing_sec:.1f}s silence padding to prevent abrupt cutoff")
                        _padded_path = work / "converted_vocals_padded.wav"
                        # afade로 마지막 0.5초 페이드아웃 + apad로 부족한 길이만큼 무음 추가
                        run_ffmpeg([
                            "-i", str(converted_vocals_path),
                            "-af", f"afade=t=out:st={max(0, _out_dur - 0.5):.2f}:d=0.5,"
                                   f"apad=pad_dur={_missing_sec:.2f}",
                            "-acodec", "pcm_s24le", "-ar", str(_process_sr),
                            str(_padded_path),
                        ])
                        if _padded_path.exists() and _padded_path.stat().st_size > 1000:
                            converted_vocals_path = _padded_path
                            log.info("Padding applied successfully")

            elif _dur_ratio > 1.15:
                log.warning(
                    f"Output longer than input: {_rvc_duration:.1f}s → {_out_dur:.1f}s"
                )
        except Exception as _vfy_err:
            log.warning(f"Output duration verification failed: {_vfy_err}")

        # --- Step 3a-fix2: 피치 아티팩트 게이트 v14 ---
        # Demucs가 악기 신호(피아노/신스 B5 ~587Hz)를 보컬 트랙에 누출시킬 때
        # RVC가 해당 신호를 A5+(>480Hz) 괴성으로 변환하는 문제 자동 수정.
        # 테너 모델 정상 한계(480Hz=B4) 초과 + 0.8초 이상 지속 구간만 감쇠.
        _artifact_gated_path = work / "converted_vocals_gated.wav"
        if _fix_pitch_artifacts(converted_vocals_path, _artifact_gated_path):
            converted_vocals_path = _artifact_gated_path

        # --- Step 3b: Post-process converted vocals for naturalness ---
        # EQ + compression + optional harmonic saturation + optional room reverb
        # Reduces AI metallic/robotic artifacts → more human-sounding result
        runpod.serverless.progress_update(job, "Post-processing vocals for naturalness... (3.5/4)")
        processed_vocals_path = work / "processed_vocals.wav"
        try:
            _post_process_vocal(
                vocal_path=converted_vocals_path,
                output_path=processed_vocals_path,
                reverb_amount=post_reverb,
                harmonic_enhance=harmonic_enhance,
                high_note_mode=high_note_mode,
                sample_rate=_process_sr,
                language=language,
            )
            if processed_vocals_path.exists() and processed_vocals_path.stat().st_size > 1000:
                converted_vocals_path = processed_vocals_path
                log.info("Vocal post-processing applied successfully")
            else:
                log.warning("Post-processing produced empty/small output, using raw conversion")
        except Exception as pp_err:
            log.warning(f"Post-processing failed, using raw RVC output: {pp_err}")

        # --- Step 3c: 원본 보컬 블렌딩 (숨결감/자연스러움 복원) v36 ---
        # RVC 변환 보컬에 원본 보컬을 vocal_blend 비율로 혼합
        # 효과: 숨소리, 자음 질감, 마이크 뉘앙스가 자연스럽게 복원됨
        if vocal_blend > 0.01 and rvc_input.exists():
            blended_path = work / "vocals_blended.wav"
            try:
                _blend_ratio = vocal_blend  # 원본 보컬 비율 (0.1 = 10%)
                run_ffmpeg([
                    "-i", str(converted_vocals_path),
                    "-i", str(rvc_input),
                    "-filter_complex",
                    f"[0:a]volume={1.0 - _blend_ratio:.3f}[conv];"
                    f"[1:a]volume={_blend_ratio:.3f}[orig];"
                    f"[conv][orig]amix=inputs=2:duration=shortest:normalize=0",
                    "-acodec", "pcm_s24le", "-ar", str(_process_sr),
                    str(blended_path),
                ])
                if blended_path.exists() and blended_path.stat().st_size > 1000:
                    converted_vocals_path = blended_path
                    log.info(f"Vocal blending applied: {_blend_ratio:.0%} original vocal mixed")
                else:
                    log.warning("Vocal blending produced empty output, skipping")
            except Exception as blend_err:
                log.warning(f"Vocal blending failed: {blend_err}, using unblended vocals")

        # --- Step 3d: 백킹 보컬 합성 (v49.6 — 화음 자연스러움) ---
        # 리드/백킹 분리된 경우: RVC 변환 리드 + 원본 백킹을 합성
        # 커뮤니티 권장: 백킹 볼륨 -3~-6dB + 고역 롤오프 → 리드 뒤로 배치
        # arca.live/postype: "같은 성별이면 백킹 원본 유지, 다른 성별이면 변환 필요"
        if backing_vocals_path and backing_vocals_path.exists():
            lead_plus_backing = work / "lead_plus_backing.wav"
            try:
                # 백킹 처리: volume 0.65 (-3.7dB) + 4kHz 이상 고역 롤오프 (-3dB)
                # 커뮤니티: "high-shelf EQ cut on backing (-2 to -4dB above 4kHz)"
                # → 백킹을 리드 뒤 soundstage로 배치 → 자연스러운 화음
                run_ffmpeg([
                    "-i", str(converted_vocals_path),
                    "-i", str(backing_vocals_path),
                    "-filter_complex",
                    f"[0:a]aformat=channel_layouts=mono[lead];"
                    f"[1:a]aformat=channel_layouts=mono,"
                    f"volume=0.65,"
                    f"highshelf=f=4000:width_type=o:width=0.7:g=-3.0[back];"
                    f"[lead][back]amix=inputs=2:duration=longest:normalize=0",
                    "-acodec", "pcm_s24le", "-ar", str(_process_sr),
                    str(lead_plus_backing),
                ])
                if lead_plus_backing.exists() and lead_plus_backing.stat().st_size > 1000:
                    converted_vocals_path = lead_plus_backing
                    log.info("Lead + backing merged (backing: -3.7dB, 4kHz shelf -3dB)")
            except Exception as lb_mix_err:
                log.warning(f"Lead/backing merge failed: {lb_mix_err}, using lead only")

        # --- Step 4: Mix converted vocals + original accompaniment ---
        mixed_path = None
        if accomp_path and accomp_path.exists():
            runpod.serverless.progress_update(job, "Mixing vocals + accompaniment... (4/4)")
            mixed_path = work / f"mixed_output.{export_format}"
            try:
                _mix_audio(converted_vocals_path, accomp_path, mixed_path,
                           vocal_volume=vocal_volume, mr_volume=mr_volume,
                           sample_rate=_process_sr)
                if mixed_path.exists():
                    log.info("Mixed output created successfully")
                else:
                    mixed_path = None
            except Exception as e:
                log.error(f"Mixing failed (returning vocals only): {e}")
                mixed_path = None

        elapsed = time.time() - start_time

        result = {
            "filename": f"converted_{Path(audio_filename).stem}.{export_format}",
            "processing_time_seconds": round(elapsed, 2),
            "parameters": {
                "pitch_shift": pitch_shift,
                "f0_method": f0_method,
                "index_rate": index_rate,
                "filter_radius": filter_radius,
                "rms_mix_rate": rms_mix_rate,
                "protect": protect,
            },
        }

        # 결과 파일을 R2에 업로드 (RunPod 응답 한도 ~20MB 초과 방지)
        # 소용량이면 inline base64, 대용량이면 R2 URL
        vocals_size = converted_vocals_path.stat().st_size
        mixed_size = mixed_path.stat().st_size if mixed_path else 0
        total_b64_size = (vocals_size + mixed_size) * 4 / 3  # base64 overhead

        if total_b64_size > 10_000_000:  # 10MB 이상이면 R2 업로드
            log.info(f"Result too large for inline ({total_b64_size / 1024 / 1024:.1f} MB b64), uploading to R2")
            try:
                from runpod.serverless.utils import upload_file_to_bucket
                bucket_name = job_input.get("bucket_name") or os.environ.get("BUCKET_NAME", "")

                vocals_url = upload_file_to_bucket(
                    file_name=f"converted_{Path(audio_filename).stem}.{export_format}",
                    file_location=str(converted_vocals_path),
                    prefix=f"convert/{job_id}",
                    bucket_name=bucket_name,
                )
                if vocals_url and vocals_url.startswith("http"):
                    result["converted_audio_url"] = vocals_url
                    log.info(f"Converted vocals uploaded to R2: {vocals_url[:80]}...")
                else:
                    # R2 미설정 → inline fallback
                    result["converted_audio"] = encode_file_b64(converted_vocals_path)

                if mixed_path:
                    mixed_url = upload_file_to_bucket(
                        file_name=f"mixed_{Path(audio_filename).stem}.{export_format}",
                        file_location=str(mixed_path),
                        prefix=f"convert/{job_id}",
                        bucket_name=bucket_name,
                    )
                    if mixed_url and mixed_url.startswith("http"):
                        result["mixed_audio_url"] = mixed_url
                        result["mixed_filename"] = f"mixed_{Path(audio_filename).stem}.{export_format}"
                        log.info(f"Mixed audio uploaded to R2: {mixed_url[:80]}...")
                    else:
                        result["mixed_audio"] = encode_file_b64(mixed_path)
                        result["mixed_filename"] = f"mixed_{Path(audio_filename).stem}.{export_format}"
            except Exception as e:
                log.warning(f"R2 upload failed, trying inline base64: {e}")
                result["converted_audio"] = encode_file_b64(converted_vocals_path)
                if mixed_path:
                    result["mixed_audio"] = encode_file_b64(mixed_path)
                    result["mixed_filename"] = f"mixed_{Path(audio_filename).stem}.{export_format}"
        else:
            # 소용량 → inline base64
            result["converted_audio"] = encode_file_b64(converted_vocals_path)
            if mixed_path:
                result["mixed_audio"] = encode_file_b64(mixed_path)
                result["mixed_filename"] = f"mixed_{Path(audio_filename).stem}.{export_format}"

        return result

    finally:
        try:
            cleanup_dir(work)
        finally:
            cleanup_gpu()


def _rvc_infer(
    pth_path: Path,
    index_path: Optional[Path],
    input_audio: Path,
    output_path: Path,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.45,     # v51: 0.55→0.45 (발음 명료도 우선)
    protect: float = 0.40,        # v49: 0.33→0.40 (과도한 자음 보호 완화 → 인덱스 정확도 향상)
    hop_length: int = 128,        # v49: 64→128 (커뮤니티 표준, 64는 노이즈 추적→삑사리)
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    export_format: str = "wav",
    filter_radius: int = 2,       # v50: 3→2 (고음 비브라토 보존)
    rms_mix_rate: float = 0.1,    # v51: 0.0→0.1 (원곡 음량 패턴 반영, 음량 균일성)
    split_audio: bool = True,
    f0_autotune: bool = True,     # v49: False→True (Applio 공식 권장: 노래 변환 시 활성)
    f0_autotune_strength: float = 0.5,  # v50: 0.3→0.5 (음정 단단하게)
) -> None:
    """
    Run RVC v2 inference using Applio's pipeline.
    Tries multiple import strategies for compatibility across Applio versions.
    IMPORTANT: Applio uses relative paths so we chdir to APPLIO_ROOT.
    """
    index_str = str(index_path) if index_path and index_path.exists() else ""
    original_cwd = os.getcwd()

    # --- Strategy 1: Applio core API ---
    try:
        os.chdir(APPLIO_ROOT)
        from core import run_infer_script

        run_infer_script(
            pitch=pitch_shift,
            filter_radius=filter_radius,
            index_rate=index_rate,
            volume_envelope=1.0,
            protect=protect,
            hop_length=hop_length,
            f0_method=f0_method,
            input_path=str(input_audio),
            output_path=str(output_path),
            pth_path=str(pth_path),
            index_path=index_str,
            split_audio=split_audio,
            f0_autotune=f0_autotune,              # v49: True (노래 변환 피치 안정화)
            f0_autotune_strength=f0_autotune_strength,  # v49.8: 0.3 (이중 스무딩→비음 완화)
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format.upper(),
            embedder_model="contentvec",
            embedder_model_custom=None,
            rms_mix_rate=rms_mix_rate,
        )
        log.info("Inference completed via core.run_infer_script")
        return
    except ImportError as e:
        log.info(f"core.run_infer_script not available ({e}), trying alternative")
    except Exception as e:
        log.warning(f"core.run_infer_script failed: {e}, trying alternative")
    finally:
        os.chdir(original_cwd)

    # --- Strategy 2: VoiceConverter class ---
    try:
        os.chdir(APPLIO_ROOT)
        from rvc.infer.infer import VoiceConverter

        vc = VoiceConverter()
        vc.convert_audio(
            audio_input_path=str(input_audio),
            audio_output_path=str(output_path),
            model_path=str(pth_path),
            index_path=index_str,
            pitch=pitch_shift,
            filter_radius=filter_radius,
            f0_method=f0_method,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            volume_envelope=1.0,
            protect=protect,
            hop_length=hop_length,
            split_audio=split_audio,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format.upper(),
            embedder_model="contentvec",
        )
        log.info("Inference completed via VoiceConverter class")
        return
    except ImportError as e:
        log.info(f"rvc.infer.infer.VoiceConverter not available ({e}), trying CLI")
    except Exception as e:
        log.warning(f"VoiceConverter failed: {e}, trying CLI")
    finally:
        os.chdir(original_cwd)

    # --- Strategy 3: Low-level pipeline fallback ---
    # If pedalboard is missing, core.py and VoiceConverter won't import.
    # Use the inference pipeline directly, bypassing the infer.py wrapper.
    net_g = None
    hubert_model = None
    pipeline = None
    try:
        os.chdir(APPLIO_ROOT)
        log.info("Attempting low-level pipeline inference (Strategy 3)")

        # Import the pipeline directly (doesn't require pedalboard)
        from rvc.infer.pipeline import Pipeline as VC
        from rvc.lib.utils import load_embedding

        import torch
        import soundfile as sf
        import numpy as np

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = device.startswith("cuda")

        # Load model
        cpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        config = cpt.get("config", [])
        tgt_sr = config[-1] if config and len(config) >= 1 else 40000
        if not tgt_sr or tgt_sr <= 1:
            tgt_sr = 40000
        # tgt_sr이 정상 샘플레이트 범위를 벗어나면 기본값 사용
        if tgt_sr not in (16000, 22050, 32000, 40000, 44100, 48000):
            log.warning(f"Suspicious tgt_sr={tgt_sr} from model config, defaulting to 40000")
            tgt_sr = 40000

        if "weight" in cpt and "emb_g.weight" in cpt["weight"] and len(config) >= 3:
            config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v2")

        from rvc.lib.algorithm.synthesizers import Synthesizer
        net_g = Synthesizer(*cpt["config"], use_f0=if_f0 == 1, text_enc_hidden_dim=768)
        del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval().to(device)
        if is_half:
            net_g = net_g.half()

        # Create pipeline
        pipeline = VC(tgt_sr, device, is_half, False)

        # Load embedding model
        models, _, _ = load_embedding("contentvec", None)
        hubert_model = models[0].to(device)
        if is_half:
            hubert_model = hubert_model.half()

        # Load audio
        audio_data, sr = sf.read(str(input_audio))
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # stereo → mono
        audio_np = audio_data.astype(np.float32)
        audio_max = np.abs(audio_np).max()
        if audio_max > 1.0:
            audio_np /= audio_max

        # Resample to 16kHz for pipeline
        import librosa
        if sr != 16000:
            audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_np

        # Load index
        file_index = ""
        if index_path and index_path.exists():
            file_index = str(index_path)

        # Run pipeline
        sid = torch.tensor([0], dtype=torch.long, device=device)
        audio_opt = pipeline.pipeline(
            hubert_model,
            net_g,
            sid,
            audio_16k,
            str(input_audio),
            [0, 0, 0],  # times (not used meaningfully)
            pitch_shift,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            0,  # resample_sr
            rms_mix_rate,
            version,
            protect,
            hop_length,
            f0_autotune,  # v49: True (노래 변환 피치 안정화)
            f0_autotune_strength,  # v49.8: 0.3 (이중 스무딩→비음 완화)
        )

        # Save output — always write as WAV first, then convert to target format if needed
        # FLOAT subtype: RVC pipeline float32 출력을 손실 없이 보존
        wav_tmp = output_path.with_suffix(".wav")
        sf.write(str(wav_tmp), audio_opt, tgt_sr, format="WAV", subtype="FLOAT")
        if output_path.suffix.lower() != ".wav":
            run_ffmpeg(["-i", str(wav_tmp), str(output_path)])
            wav_tmp.unlink(missing_ok=True)
        elif wav_tmp != output_path:
            wav_tmp.rename(output_path)
        log.info(f"Inference completed via low-level pipeline (output: {output_path.stat().st_size / 1024:.1f} KB)")
        return
    except Exception as e:
        log.warning(f"Low-level pipeline failed: {e}", exc_info=True)
    finally:
        os.chdir(original_cwd)
        # Strategy 3 GPU 모델 정리 (Strategy 4를 위해 VRAM 확보)
        del net_g, hubert_model, pipeline
        cleanup_gpu()

    # --- Strategy 4: CLI fallback (last resort) ---
    cmd = [
        sys.executable,
        str(APPLIO_ROOT / "core.py"),
        "infer",
        "--pitch", str(pitch_shift),
        "--filter_radius", str(filter_radius),
        "--index_rate", str(index_rate),
        "--rms_mix_rate", str(rms_mix_rate),
        "--volume_envelope", "1.0",
        "--protect", str(protect),
        "--hop_length", str(hop_length),
        "--f0_method", f0_method,
        "--input_path", str(input_audio),
        "--output_path", str(output_path),
        "--pth_path", str(pth_path),
        "--index_path", index_str,
        "--split_audio", str(split_audio),
        "--clean_audio", str(clean_audio),
        "--clean_strength", str(clean_strength),
        "--export_format", export_format.upper(),
        "--embedder_model", "contentvec",
        "--f0_autotune", str(f0_autotune),
        "--f0_autotune_strength", str(f0_autotune_strength),
    ]

    log.info(f"CLI command: {' '.join(cmd)}")
    cli_result = subprocess.run(
        cmd,
        cwd=str(APPLIO_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    log.info(f"CLI exit code: {cli_result.returncode}")
    if cli_result.stdout.strip():
        log.info(f"CLI stdout: {cli_result.stdout[-2000:]}")
    if cli_result.stderr.strip():
        log.warning(f"CLI stderr: {cli_result.stderr[-2000:]}")

    if cli_result.returncode != 0:
        # Include last 5 lines of stdout for debugging context
        stdout_lines = [ln for ln in cli_result.stdout.strip().splitlines() if ln.strip()]
        tail = "\n".join(stdout_lines[-5:]) if stdout_lines else "(no stdout)"
        stderr_tail = cli_result.stderr.strip()[-500:] if cli_result.stderr.strip() else ""
        raise RuntimeError(
            f"All 4 RVC inference strategies failed. CLI exit code: {cli_result.returncode}.\n"
            f"Last stdout:\n{tail}"
            + (f"\nStderr:\n{stderr_tail}" if stderr_tail else "")
        )
    else:
        log.info("Inference completed via CLI fallback")

    # --- 출력 파일 탐색: CLI가 다른 경로에 저장했을 수 있음 ---
    if not output_path.exists():
        log.warning(f"Output not at expected path: {output_path}")
        # work 디렉토리에서 변환된 파일 검색
        found_files = list(output_path.parent.glob("*"))
        log.info(f"Files in work dir: {[f.name for f in found_files]}")
        # Applio 기본 출력 디렉토리도 검색
        for search_dir in [
            APPLIO_ROOT / "audio" / "outputs",
            APPLIO_ROOT / "audio",
            APPLIO_ROOT / "output",
            APPLIO_ROOT / "outputs",
        ]:
            if search_dir.exists():
                applio_files = list(search_dir.rglob(f"*.{export_format}"))
                if applio_files:
                    log.info(f"Found in {search_dir}: {[f.name for f in applio_files]}")
                    # 가장 최근 파일을 사용
                    newest = max(applio_files, key=lambda f: f.stat().st_mtime)
                    shutil.copy2(str(newest), str(output_path))
                    log.info(f"Copied {newest} → {output_path}")
                    break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main RunPod Handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handler(job: dict) -> dict:
    """
    RunPod Serverless handler entry point.

    Expected input format:
    {
      "input": {
        "task_type": "preprocess" | "train" | "convert",
        ... task-specific fields ...
      }
    }
    """
    job_input: dict = job.get("input", {})
    task_type: str = job_input.get("task_type", "")
    job_id: str = job.get("id", "unknown")

    log.info(f"=== Job {job_id} started: task_type={task_type} ===")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Ensure work directory exists
    ensure_dir(WORK_DIR)

    start = time.time()

    try:
        if task_type == "preprocess":
            result = task_preprocess(job_input, job)
        elif task_type == "train":
            result = task_train(job_input, job)
        elif task_type == "convert":
            result = task_convert(job_input, job)
        else:
            raise ValueError(
                f"Unknown task_type: '{task_type}'. "
                f"Supported: preprocess, train, convert"
            )

        elapsed = time.time() - start
        log.info(f"=== Job {job_id} completed in {elapsed:.1f}s ===")
        return result

    except Exception as e:
        elapsed = time.time() - start
        error_msg = f"{type(e).__name__}: {str(e)}"
        log.error(f"=== Job {job_id} failed after {elapsed:.1f}s: {error_msg} ===")
        log.error(traceback.format_exc())
        cleanup_gpu()

        # Return error in RunPod's expected format
        raise RuntimeError(error_msg) from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entrypoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    try:
        log.info("AI Voice Studio - RunPod Handler starting")
        log.info(f"Python: {sys.version}")
        log.info(f"Applio root: {APPLIO_ROOT} (exists: {APPLIO_ROOT.exists()})")

        # Applio 디렉토리 구조 확인
        if APPLIO_ROOT.exists():
            core_py = APPLIO_ROOT / "core.py"
            rvc_dir = APPLIO_ROOT / "rvc"
            log.info(f"  core.py exists: {core_py.exists()}")
            log.info(f"  rvc/ exists: {rvc_dir.exists()}")
            if rvc_dir.exists():
                subdirs = [d.name for d in rvc_dir.iterdir() if d.is_dir()]
                log.info(f"  rvc/ subdirs: {subdirs}")

        log.info(f"Work directory: {WORK_DIR}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 핵심 모듈 import 테스트
        import_tests = [
            ("numpy", "numpy"),
            ("soundfile", "soundfile"),
            ("librosa", "librosa"),
            ("demucs", "demucs"),
            ("noisereduce", "noisereduce"),
            ("faiss", "faiss"),
        ]
        for name, mod in import_tests:
            try:
                __import__(mod)
                log.info(f"  {name}: OK")
            except ImportError as e:
                log.warning(f"  {name}: MISSING ({e})")

        log.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})

    except Exception as e:
        log.error(f"FATAL: Handler startup failed: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        # stderr에도 출력 (RunPod 로그에 확실히 남도록)
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
