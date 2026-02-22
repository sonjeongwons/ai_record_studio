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
  - FFmpeg, Demucs (htdemucs_ft), pyannote-audio, noisereduce
  - CUDA 12.1+, PyTorch 2.x, FAISS-gpu
  - runpod SDK
  - pretrained models at /app/Applio/rvc/models/pretraineds/
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import uuid
import shutil
import base64
import logging
import traceback
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Optional

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

# Segment duration bounds (seconds)
SEGMENT_MIN = 5.0
SEGMENT_MAX = 15.0

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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def decode_b64_file(data_b64: str, dest_path: Path) -> Path:
    """Decode a base64-encoded string and write it to dest_path."""
    raw = base64.b64decode(data_b64)
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
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def run_ffmpeg(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with timeout."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    log.info(f"FFmpeg: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PREPROCESS task
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def task_preprocess(job_input: dict, job: dict) -> dict:
    """
    Full preprocessing pipeline:
      1) Decode incoming audio/video files
      2) Extract audio from video (FFmpeg)
      3) Vocal separation (Demucs htdemucs_ft)
      4) Speaker diarization (pyannote, optional)
      5) Noise reduction (noisereduce)
      6) Segment into 5-15s clips
      7) Return segments as base64 + metadata
    """
    audio_files: list[dict] = job_input.get("audio_files", [])
    if not audio_files:
        raise ValueError("No audio_files provided for preprocessing")

    job_id = job.get("id", uuid.uuid4().hex[:12])
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
            fname = fobj.get("filename", f"input_{i}.wav")
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

        # --- Step 3: Vocal separation with Demucs htdemucs_ft ---
        runpod.serverless.progress_update(job, "Separating vocals (Demucs)... (3/6)")
        separation = _demucs_separate(audio_paths, vocal_dir)
        vocal_paths = separation["vocals"]
        accomp_paths = separation["accompaniment"]

        # --- Step 4: Speaker diarization (optional) ---
        runpod.serverless.progress_update(job, "Speaker diarization... (4/6)")
        diarized_paths = _speaker_diarize(vocal_paths, work / "diarized")

        # --- Step 5: Noise reduction ---
        runpod.serverless.progress_update(job, "Noise reduction... (5/6)")
        cleaned_paths = _noise_reduce(diarized_paths, clean_dir)

        # --- Step 6: Segment into 5-15s clips ---
        runpod.serverless.progress_update(job, "Segmenting audio clips... (6/6)")
        segments = _segment_audio(cleaned_paths, segment_dir)

        # Convert segments to MP3 for transfer
        segment_data = []
        total_duration = 0.0
        mp3_dir = ensure_dir(work / "mp3_transfer")

        for seg_path in sorted(segment_dir.glob("*.wav")):
            dur = get_audio_duration(seg_path)
            total_duration += dur

            # Convert WAV → MP3 192kbps mono for smaller transfer
            mp3_path = mp3_dir / (seg_path.stem + ".mp3")
            try:
                run_ffmpeg([
                    "-i", str(seg_path),
                    "-codec:a", "libmp3lame", "-b:a", "192k",
                    "-ar", "44100", "-ac", "1",
                    str(mp3_path),
                ])
                encode_path = mp3_path
            except Exception as e:
                log.warning(f"MP3 encode failed for {seg_path.name}: {e}, using WAV")
                encode_path = seg_path

            segment_data.append({
                "filename": encode_path.name,
                "path": str(encode_path),
                "duration_seconds": round(dur, 2),
            })

        # Compress accompaniment files to MP3 for download (stereo, high quality)
        accomp_data = []
        for accomp_path in accomp_paths:
            dur = get_audio_duration(accomp_path)
            mp3_path = mp3_dir / f"mr_{accomp_path.stem}.mp3"
            try:
                run_ffmpeg([
                    "-i", str(accomp_path),
                    "-codec:a", "libmp3lame", "-b:a", "320k",
                    "-ar", "44100", "-ac", "2",
                    str(mp3_path),
                ])
                encode_path = mp3_path
            except Exception:
                encode_path = accomp_path
            accomp_data.append({
                "filename": encode_path.name,
                "path": str(encode_path),
                "duration_seconds": round(dur, 2),
            })

        # Compress full vocal files (pre-diarization) for download
        vocal_data = []
        for vp in vocal_paths:
            dur = get_audio_duration(vp)
            mp3_path = mp3_dir / f"vocal_{vp.stem}.mp3"
            try:
                run_ffmpeg([
                    "-i", str(vp),
                    "-codec:a", "libmp3lame", "-b:a", "192k",
                    "-ar", "44100", "-ac", "1",
                    str(mp3_path),
                ])
                encode_path = mp3_path
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
            bucket_name = os.environ.get("BUCKET_NAME", "voice-studio")
            # Test if bucket is configured by checking env vars
            if os.environ.get("BUCKET_ENDPOINT_URL"):
                use_r2 = True
        except ImportError:
            pass

        if use_r2:
            log.info(f"Uploading {len(all_files)} preprocessed files to R2...")
            for fobj in all_files:
                try:
                    url = upload_file_to_bucket(
                        file_name=fobj["filename"],
                        file_location=fobj["path"],
                        prefix=f"voice-studio/preprocess/{job_id}",
                        bucket_name=bucket_name,
                    )
                    if url and url.startswith("http"):
                        fobj["url"] = url
                    else:
                        # No bucket configured — fall back to inline base64
                        fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))
                except Exception as e:
                    log.warning(f"R2 upload failed for {fobj['filename']}: {e}, using base64")
                    fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))
            log.info(f"R2 upload complete for preprocess results")
        else:
            # No R2 configured — use inline base64 (may exceed payload limit)
            log.warning("No R2 bucket configured — using inline base64 for preprocess results. "
                        "This may fail for large datasets due to RunPod payload limits.")
            for fobj in all_files:
                fobj["data_base64"] = encode_file_b64(Path(fobj["path"]))

        # Remove local paths from response (not needed by server)
        for fobj in all_files:
            fobj.pop("path", None)

        return {
            "segment_count": len(segment_data),
            "total_duration": round(total_duration, 2),
            "segments": segment_data,
            "accompaniment_files": accomp_data,
            "vocal_files": vocal_data,
        }

    finally:
        cleanup_dir(work)
        cleanup_gpu()


def _demucs_separate(audio_paths: list[Path], output_dir: Path) -> dict:
    """
    Run Demucs htdemucs_ft model to separate vocals from accompaniment.
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
            if peak > 0.9:  # normalize only if near clipping
                result = result * (0.9 / peak)
        return result

    # --- Try demucs.api first (GitHub v4.1.0a2+) ---
    try:
        import demucs.api
        log.info("Using demucs.api (v4.1+) for vocal separation (shifts=5, overlap=0.5)")
        separator = demucs.api.Separator(
            model="htdemucs_ft",
            device=device,
            shifts=5,       # 5 random time shifts → average = dramatically better SDR
            overlap=0.5,    # 50% overlap between segments = smoother transitions
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
                    sf.write(str(vocal_wav), vocal_np, samplerate=44100, subtype="PCM_16")
                    vocal_paths.append(vocal_wav)

                    # Build accompaniment = drums + bass + other (STEREO preserved)
                    mr_stems = []
                    for stem in ("drums", "bass", "other"):
                        if stem in separated:
                            mr_stems.append(separated[stem].cpu().numpy())
                    if mr_stems:
                        accomp_np = _safe_sum_stems(mr_stems)
                        accomp_wav = output_dir / f"{audio_path.stem}_accompaniment.wav"
                        # Transpose [channels, samples] → [samples, channels] for soundfile
                        sf.write(str(accomp_wav), accomp_np.T, samplerate=44100, subtype="PCM_16")
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

    model = get_model("htdemucs_ft")
    model.to(device)
    # htdemucs_ft sources order: drums, bass, other, vocals
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
            wav = (wav - ref.mean()) / ref.std()
            sources = apply_model(model, wav[None], device=device, shifts=5, overlap=0.5)[0]
            sources = sources * ref.std() + ref.mean()

            if vocals_idx >= 0:
                # Save vocals as MONO (RVC requires mono input)
                vocal_wav_path = output_dir / f"{audio_path.stem}_vocals.wav"
                vocal_np = _to_mono(sources[vocals_idx].cpu().numpy())
                sf.write(str(vocal_wav_path), vocal_np, samplerate=model.samplerate, subtype="PCM_16")
                vocal_paths.append(vocal_wav_path)

                # Build accompaniment: sum all non-vocal sources (STEREO preserved)
                mr_stems = []
                for i, sname in enumerate(source_names):
                    if sname != "vocals":
                        mr_stems.append(sources[i].cpu().numpy())
                if mr_stems:
                    accomp_np = _safe_sum_stems(mr_stems)
                    accomp_wav_path = output_dir / f"{audio_path.stem}_accompaniment.wav"
                    sf.write(str(accomp_wav_path), accomp_np.T, samplerate=model.samplerate, subtype="PCM_16")
                    accomp_paths.append(accomp_wav_path)

                log.info(f"Demucs separation done (fallback): {audio_path.name} → vocals (mono) + accompaniment (stereo)")
            else:
                log.warning(f"No 'vocals' source in model for {audio_path.name}, using original")
                vocal_paths.append(audio_path)
        except Exception as e:
            log.error(f"Demucs fallback failed for {audio_path.name}: {e}")
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
                sf.write(str(out_path), concatenated, samplerate=sr, subtype="PCM_16")
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
    Light noise reduction using noisereduce library.
    Optimized for singing vocals after Demucs separation:
    - prop_decrease=0.4 (gentle) to preserve high-frequency detail
      (consonants, sibilants, breath sounds that give voice character)
    - Demucs already removes most background noise, so aggressive
      noise reduction here degrades quality rather than improving it
    - Only targets residual stationary noise (hum, hiss)
    """
    ensure_dir(output_dir)

    try:
        import noisereduce as nr
        import soundfile as sf
        import numpy as np

        cleaned: list[Path] = []
        for ap in audio_paths:
            audio_data, sr = sf.read(str(ap))

            # Check if audio is already clean (Demucs output usually is)
            # RMS below -50dB noise floor → skip noise reduction entirely
            rms = np.sqrt(np.mean(audio_data ** 2))
            noise_floor_db = 20 * np.log10(max(rms, 1e-10))

            if noise_floor_db < -50:
                # Extremely quiet / already very clean → skip
                log.info(f"Audio already clean ({noise_floor_db:.1f}dB), skipping NR: {ap.name}")
                cleaned.append(ap)
                continue

            # Gentle noise reduction — preserve singing detail
            reduced = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                prop_decrease=0.4,     # gentle (was 0.8, too aggressive for singing)
                stationary=True,
                n_fft=2048,
                hop_length=512,
            )

            out_path = output_dir / f"{ap.stem}_clean.wav"
            sf.write(str(out_path), reduced, samplerate=sr, subtype="PCM_16")
            cleaned.append(out_path)
            log.info(f"Noise reduction done (gentle): {ap.name}")

        return cleaned

    except ImportError:
        log.warning("noisereduce not installed, skipping noise reduction")
        return audio_paths
    except Exception as e:
        log.error(f"Noise reduction failed: {e}")
        return audio_paths


def _segment_audio(audio_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    Split audio files into 5-15 second segments using silence detection.
    Uses FFmpeg silencedetect to find natural split points,
    falling back to fixed-length splitting.
    """
    import soundfile as sf

    ensure_dir(output_dir)
    all_segments: list[Path] = []
    seg_idx = 0

    for ap in audio_paths:
        # 원본 파일명에서 처리 접미사들을 모두 제거하여 소스 stem 추출
        # 순서 중요: _diarized → _clean → _vocals (역순으로 붙었으므로)
        source_stem = ap.stem
        for suffix in ("_diarized", "_clean", "_vocals"):
            if source_stem.endswith(suffix):
                source_stem = source_stem[:-len(suffix)]

        audio_data, sr = sf.read(str(ap))
        total_samples = len(audio_data)
        total_duration = total_samples / sr

        if total_duration <= SEGMENT_MAX:
            # File is already short enough, keep as-is if >= SEGMENT_MIN
            if total_duration >= SEGMENT_MIN:
                out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
                sf.write(str(out), audio_data, samplerate=sr, subtype="PCM_16")
                all_segments.append(out)
                seg_idx += 1
            elif total_duration >= 2.0:
                # Keep very short clips too (will be less than 5s but still usable)
                out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
                sf.write(str(out), audio_data, samplerate=sr, subtype="PCM_16")
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
                if (end - pos) >= 2.0:
                    split_points.append((pos, end))
                pos = end

        for start_sec, end_sec in split_points:
            start_s = int(start_sec * sr)
            end_s = min(int(end_sec * sr), total_samples)
            segment = audio_data[start_s:end_s]

            out = output_dir / f"{source_stem}_seg_{seg_idx:04d}.wav"
            sf.write(str(out), segment, samplerate=sr, subtype="PCM_16")
            all_segments.append(out)
            seg_idx += 1

    log.info(f"Total segments created: {len(all_segments)}")
    return all_segments


def _detect_silence_splits(
    audio_path: str, sr: int, total_duration: float
) -> list[tuple[float, float]]:
    """
    Use FFmpeg silencedetect to find silence boundaries,
    then create segments of 5-15 seconds at those boundaries.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", audio_path,
                "-af", "silencedetect=noise=-35dB:d=0.3",
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
    sample_rate: int = job_input.get("sample_rate", 40000)  # 40k recommended for SVC
    epochs: int = job_input.get("epochs", 500)
    batch_size: int = job_input.get("batch_size", 0)  # 0 = auto-detect
    f0_method: str = job_input.get("f0_method", "rmvpe")
    embedder_model: str = job_input.get("embedder_model", "contentvec")
    save_every_epoch: int = job_input.get("save_every_epoch", 50)

    if not audio_files and not audio_urls:
        raise ValueError("No audio_files or audio_urls provided for training")

    # Validate model name — must be safe for filesystem paths
    import re
    model_name = re.sub(r'[<>:"/\\|?*]', '_', model_name).strip()
    if not model_name:
        model_name = "my_voice_model"

    # Validate parameters — Applio only supports 32k, 40k, 48k
    if sample_rate not in (32000, 40000, 48000):
        log.warning(f"Invalid sample_rate {sample_rate}, defaulting to 40000")
        sample_rate = 40000
    epochs = max(1, min(epochs, 10000))

    # Auto-detect optimal batch_size from GPU VRAM if not specified
    if batch_size <= 0:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            # Rule of thumb: batch_size ≈ VRAM_GB, in multiples of 4
            # RTX 4090 (24GB) → 16~24, RTX 3090 (24GB) → 16, RTX 3060 (12GB) → 8
            batch_size = max(4, min(32, int(vram_gb) // 4 * 4))
            log.info(f"Auto-detected batch_size={batch_size} for {vram_gb:.1f}GB VRAM")
        except Exception:
            batch_size = 8
    batch_size = max(4, min(32, batch_size))

    job_id = job.get("id", uuid.uuid4().hex[:12])
    start_time = time.time()

    # Applio expects a specific directory structure for training:
    #   logs/{model_name}/       - training logs, checkpoints
    #   logs/{model_name}/1_2_3/ - preprocessed sliced audio
    #   logs/{model_name}/0_gt_wavs/ - preprocessed audio
    #   logs/{model_name}/3_feature768/ or 3_feature256/ - feature files
    logs_dir = APPLIO_ROOT / "logs" / model_name
    dataset_dir = ensure_dir(WORK_DIR / f"train_{job_id}" / "dataset")
    ensure_dir(logs_dir)

    # Map sample rate to Applio's internal version string
    sr_map = {32000: "32k", 40000: "40k", 48000: "48k"}
    sr_label = sr_map.get(sample_rate, "40k")

    try:
        # --- Step 1: Download/decode audio files to dataset directory ---
        total_files = len(audio_files) + len(audio_urls)
        runpod.serverless.progress_update(job, f"Downloading {total_files} audio files... (1/5)")

        # R2 URL에서 다운로드
        for i, uobj in enumerate(audio_urls):
            fname = uobj.get("filename", f"audio_{i}.wav")
            dest = dataset_dir / fname
            import requests as _req
            resp = _req.get(uobj["url"], timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            log.info(f"Downloaded training file: {fname} ({len(resp.content) / 1024:.1f} KB)")

        # base64 인라인 디코딩 (하위 호환)
        for i, fobj in enumerate(audio_files):
            fname = fobj.get("filename", f"audio_{i}.wav")
            dest = dataset_dir / fname
            decode_b64_file(fobj["data_base64"], dest)
            log.info(f"Decoded training file: {fname}")

        # Convert all to WAV at target sample rate
        wav_dir = ensure_dir(WORK_DIR / f"train_{job_id}" / "wav")
        for f in dataset_dir.iterdir():
            if f.suffix.lower() in AUDIO_EXTS:
                out_wav = wav_dir / (f.stem + ".wav")
                run_ffmpeg([
                    "-i", str(f),
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
            raise RuntimeError(f"[2/6 전처리] {e}") from e

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

        result = {
            "model_name": model_name,
            "epochs_trained": epochs,
            "sample_rate": sample_rate,
            "training_time_seconds": round(elapsed, 1),
            "pth_filename": pth_path.name,
        }

        try:
            from runpod.serverless.utils import upload_file_to_bucket

            bucket_name = os.environ.get("BUCKET_NAME", "voice-studio")

            pth_url = upload_file_to_bucket(
                file_name=pth_path.name,
                file_location=str(pth_path),
                prefix=f"voice-studio/{job_id}",
                bucket_name=bucket_name,
            )
            # upload_file_to_bucket returns a local path if no bucket configured
            if pth_url and pth_url.startswith("http"):
                result["pth_url"] = pth_url
                log.info(f"Model uploaded to bucket: {pth_url[:80]}...")
            else:
                raise RuntimeError(
                    "클라우드 스토리지가 설정되지 않았습니다. "
                    "RunPod 템플릿에 BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, "
                    "BUCKET_SECRET_ACCESS_KEY 환경변수를 설정하세요. "
                    "(Cloudflare R2 무료 플랜 권장)"
                )

            if index_path is not None:
                index_url = upload_file_to_bucket(
                    file_name=index_path.name,
                    file_location=str(index_path),
                    prefix=f"voice-studio/{job_id}",
                    bucket_name=bucket_name,
                )
                if index_url and index_url.startswith("http"):
                    result["index_url"] = index_url
                    result["index_filename"] = index_path.name
                    log.info(f"Index uploaded to bucket: {index_url[:80]}...")
            else:
                log.warning("No FAISS index file generated")

        except ImportError:
            raise RuntimeError(
                "boto3가 설치되지 않았습니다. Docker 이미지를 재빌드하세요."
            )
        except RuntimeError:
            raise  # re-raise our own errors
        except Exception as e:
            raise RuntimeError(
                f"모델 업로드 실패: {e}. "
                f"RunPod 환경변수(BUCKET_ENDPOINT_URL 등)를 확인하세요."
            ) from e

        return result

    finally:
        cleanup_dir(WORK_DIR / f"train_{job_id}")
        # logs_dir contains the model, but we've already uploaded/encoded it above,
        # so it's safe to clean. However, keep logs_dir if upload failed to allow retry.
        # Only clean non-essential subdirectories, preserve .pth and .index
        try:
            if logs_dir.exists():
                for sub in logs_dir.iterdir():
                    if sub.is_dir():
                        cleanup_dir(sub)
                    elif sub.suffix not in (".pth", ".index"):
                        sub.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Partial logs cleanup failed: {e}")
        cleanup_gpu()


def _rvc_preprocess(
    model_name: str, dataset_path: str, sample_rate: int, logs_dir: Path
) -> None:
    """
    RVC preprocessing: slice, resample and organize audio for training.

    Self-contained implementation using FFmpeg — avoids Applio's import chain
    (core.py → launch_tensorboard → tensorboard, etc.) which frequently breaks.

    Steps:
      1) Slice each audio file into ~1.5s segments (Applio's expected segment size)
      2) Resample segments to target sample rate → sliced_audios/
      3) Resample segments to 16kHz → sliced_audios_16k/

    Creates the directory structure current Applio expects:
      - logs/{model_name}/sliced_audios/     → sliced audio at target sample rate
      - logs/{model_name}/sliced_audios_16k/ → sliced audio at 16kHz (for extraction)

    Also creates config.json from rvc/configs/{sample_rate}.json template.
    """
    SLICE_DURATION = 1.5  # seconds — Applio's standard segment size

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

    idx = 0
    for audio_file in audio_files:
        try:
            duration = get_audio_duration(audio_file)
            if duration < 0.5:
                log.warning(f"Skipping too-short file ({duration:.1f}s): {audio_file.name}")
                continue

            if duration <= SLICE_DURATION * 1.5:
                # Short file — use as single segment (no slicing needed)
                slice_paths = [audio_file]
            else:
                # Slice into ~1.5s segments using FFmpeg segment muxer
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
                    continue  # skip very short tail segments

                # CRITICAL: Prefix with "0_" so Applio's generate_filelist()
                # assigns speaker_id=0 to ALL segments (single-speaker training).
                # Applio extracts sid via: name.split("_")[0]
                # Without prefix, "0000001" → sid=1 → multiple speaker IDs
                # → CUDA assert when segment count > spk_embed_dim (109)
                padded = f"0_{idx:07d}"

                # Save at target sample rate (mono, 16-bit PCM, soxr HQ resampler)
                gt_path = gt_dir / f"{padded}.wav"
                run_ffmpeg([
                    "-i", str(sp),
                    "-af", f"aresample=resampler=soxr:precision=28:osr={sample_rate}",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(gt_path),
                ])

                # Save at 16kHz for F0 & feature extraction (matching naming)
                sr16k_path = sr16k_dir / f"{padded}.wav"
                run_ffmpeg([
                    "-i", str(sp),
                    "-af", "aresample=resampler=soxr:precision=28:osr=16000",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(sr16k_path),
                ])

                idx += 1

        except Exception as e:
            log.warning(f"Failed to preprocess {audio_file.name}: {e}")
            continue

    # Cleanup temp slices
    cleanup_dir(tmp_slices)

    if idx == 0:
        raise RuntimeError("전처리에 성공한 오디오 파일이 없습니다")

    log.info(f"RVC preprocess completed: {idx} sliced segments → {gt_dir} + {sr16k_dir}")

    # Create config.json from Applio's sample-rate template
    config_src = APPLIO_ROOT / "rvc" / "configs" / f"{sample_rate}.json"
    config_dst = logs_dir / "config.json"
    if config_src.exists() and not config_dst.exists():
        import shutil
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
    for line in iter(proc.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue
        log.info(f"[extract] {line}")
        last_lines.append(line)
        if len(last_lines) > 20:
            last_lines.pop(0)

    proc.wait(timeout=7200)
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
            import shutil
            shutil.copyfile(str(config_src), str(config_json))
    if not filelist_txt.exists():
        log.warning("extract.py did not create filelist.txt — training may fail")
    else:
        # Verify filelist has actual training data (not just mute entries)
        with open(filelist_txt, "r") as f:
            lines = [l.strip() for l in f if l.strip() and "mute" not in l.lower()]
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
) -> dict:
    """
    Run RVC v2 training loop via Applio CLI.
    Uses CLI (subprocess) instead of core API because core API
    doesn't check subprocess return codes.
    """
    # Determine pretrained model paths
    pretrained_g = _find_pretrained("G", sr_label)
    pretrained_d = _find_pretrained("D", sr_label)

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
        runpod.serverless.progress_update(job, f"⚠ 사전학습 모델 없이 학습 시작 (품질 저하 가능)")

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

    # Also verify filelist.txt exists and has entries
    filelist_txt = logs_dir / "filelist.txt"
    if filelist_txt.exists():
        with open(filelist_txt, "r") as f:
            fl_lines = [l.strip() for l in f if l.strip() and "mute" not in l.lower()]
        log.info(f"filelist.txt before training: {len(fl_lines)} entries")
        if not fl_lines:
            raise RuntimeError(
                "filelist.txt is empty (no training data entries). "
                "Check that preprocessing and extraction produced matching files."
            )
    else:
        log.error(f"filelist.txt not found at {filelist_txt} — training will likely fail")

    # Call train.py DIRECTLY — bypassing core.py which swallows subprocess errors.
    # train.py uses positional sys.argv arguments in this exact order:
    #   1: model_name, 2: save_every_epoch, 3: total_epoch,
    #   4: pretrainG, 5: pretrainD, 6: gpu, 7: batch_size, 8: sample_rate,
    #   9: save_only_latest, 10: save_every_weights, 11: cache_data_in_gpu,
    #   12: overtraining_detector, 13: overtraining_threshold, 14: cleanup,
    #   15: vocoder, 16: checkpointing
    pg = str(pretrained_g) if pretrained_g else ""
    pd = str(pretrained_d) if pretrained_d else ""
    cmd = [
        sys.executable,
        str(APPLIO_ROOT / "rvc" / "train" / "train.py"),
        model_name,               # 1
        str(save_every_epoch),    # 2
        str(epochs),              # 3
        pg,                       # 4: pretrainG path
        pd,                       # 5: pretrainD path
        "0",                      # 6: gpu
        str(batch_size),          # 7
        str(sample_rate),         # 8
        "True",                   # 9: save_only_latest (avoid G_/D_ checkpoint bloat)
        "True",                   # 10: save_every_weights
        "False",                  # 11: cache_data_in_gpu
        "True",                   # 12: overtraining_detector
        "75",                     # 13: overtraining_threshold
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
        import re as _re
        prev_epoch = epoch_count
        epoch_match = _re.search(r'epoch[=:\s]+(\d+)', line, _re.IGNORECASE)
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

    proc.wait(timeout=36000)  # 10-hour timeout
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


def _find_pretrained(gen_or_disc: str, sr_label: str) -> Optional[Path]:
    """
    Find pretrained generator/discriminator model for transfer learning.
    Applio stores them under rvc/models/pretraineds/pretrained_v2/
    """
    search_dirs = [
        PRETRAINED_DIR / "pretrained_v2",
        PRETRAINED_DIR,
        APPLIO_ROOT / "assets" / "pretrained_v2",
        APPLIO_ROOT / "rvc" / "pretraineds" / "pretrained_v2",
    ]

    # Standard naming: f0G48k.pth, f0D48k.pth (with pitch guidance)
    # or G48k.pth, D48k.pth (without pitch guidance)
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
                log.info(f"Found pretrained {gen_or_disc}: {candidate}")
                return candidate

    log.warning(f"No pretrained {gen_or_disc} model found for {sr_label}")
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
            model_name=model_name,
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

        feature_dir = logs_dir / "3_feature768"
        if not feature_dir.exists():
            feature_dir = logs_dir / "3_feature256"

        if not feature_dir.exists():
            log.warning(f"Feature directory not found: {feature_dir}")
            return

        # Collect all feature .npy files
        npys = sorted(feature_dir.glob("*.npy"))
        if not npys:
            log.warning("No .npy feature files found for index creation")
            return

        # Stack all feature vectors
        features = []
        for npy_file in npys:
            feat = np.load(str(npy_file))
            features.append(feat)
        big_npy = np.concatenate(features, axis=0).astype(np.float32)

        log.info(f"Building FAISS index from {len(npys)} files, {big_npy.shape[0]} vectors")

        # Save concatenated features
        big_npy_path = logs_dir / "total_fea.npy"
        np.save(str(big_npy_path), big_npy)

        # Build IVF index
        n_ivf = min(int(big_npy.shape[0] ** 0.5), big_npy.shape[0])
        n_ivf = max(1, n_ivf)

        dim = big_npy.shape[1]  # 768 for ContentVec, 256 for HuBERT
        index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")

        # Train index
        index.train(big_npy)
        index.add(big_npy)

        index_path = logs_dir / f"{model_name}.index"
        faiss.write_index(index, str(index_path))
        log.info(f"FAISS index created: {index_path} ({big_npy.shape[0]} vectors)")

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

def _mix_audio(
    vocal_path: Path,
    accomp_path: Path,
    output_path: Path,
    vocal_volume: float = 1.0,
    mr_volume: float = 1.0,
) -> None:
    """Mix converted vocals with original accompaniment (MR) using FFmpeg.

    Key: amix normalize=0 prevents the default behavior of dividing output
    by number of inputs (which halves volume). alimiter prevents clipping
    after summation while preserving dynamics.
    Vocal (mono) is auto-upmixed to stereo by amix to match MR channels.
    """
    run_ffmpeg([
        "-i", str(vocal_path),
        "-i", str(accomp_path),
        "-filter_complex",
        f"[0:a]aresample=resampler=soxr,volume={vocal_volume}[v];"
        f"[1:a]aresample=resampler=soxr,volume={mr_volume}[m];"
        f"[v][m]amix=inputs=2:duration=longest:normalize=0,"
        f"alimiter=limit=0.99:attack=5:release=50",
        "-acodec", "pcm_s24le",  # 24-bit for maximum dynamic range
        "-ar", "44100",
        str(output_path),
    ])
    log.info(f"Mixed audio: {output_path.name} (vocal={vocal_volume}, mr={mr_volume})")


def task_convert(job_input: dict, job: dict) -> dict:
    """
    RVC v2 voice conversion (SVC pipeline):
      1) Decode model (.pth, .index) and input audio
      2) Demucs vocal separation → vocals + accompaniment (MR)
      3) RVC voice conversion on vocals only
      4) Mix converted vocals + original MR
      5) Return converted vocals + mixed output as base64
    """
    pth_b64: str = job_input.get("pth_data", "")
    pth_url: str = job_input.get("pth_url", "")
    index_b64: str = job_input.get("index_data", "")
    index_url: str = job_input.get("index_url", "")
    audio_b64: str = job_input.get("audio_data", "")
    audio_url: str = job_input.get("audio_url", "")
    audio_filename: str = job_input.get("audio_filename", "input.wav")
    pitch_shift: int = job_input.get("pitch_shift", 0)
    index_rate: float = job_input.get("index_rate", 0.88)
    f0_method: str = job_input.get("f0_method", "rmvpe")
    filter_radius: int = job_input.get("filter_radius", 3)
    rms_mix_rate: float = job_input.get("rms_mix_rate", 0.1)
    protect: float = job_input.get("protect", 0.23)
    hop_length: int = job_input.get("hop_length", 128)
    clean_audio: bool = job_input.get("clean_audio", True)
    clean_strength: float = job_input.get("clean_strength", 0.7)
    export_format: str = job_input.get("export_format", "wav")
    # SVC pipeline options
    separate_vocals: bool = job_input.get("separate_vocals", True)
    vocal_volume: float = job_input.get("vocal_volume", 1.0)
    mr_volume: float = job_input.get("mr_volume", 1.0)

    if not pth_b64 and not pth_url:
        raise ValueError("No model provided (pth_data or pth_url required)")
    if not audio_b64 and not audio_url:
        raise ValueError("No audio provided (audio_data or audio_url required)")

    job_id = job.get("id", uuid.uuid4().hex[:12])
    work = ensure_dir(WORK_DIR / f"convert_{job_id}")
    start_time = time.time()

    try:
        # --- Step 1: Decode files ---
        runpod.serverless.progress_update(job, "Decoding model and audio files... (1/4)")

        # Model: download from URL or decode base64
        if pth_url:
            import requests as _req
            log.info(f"Downloading model from URL: {pth_url[:80]}...")
            resp = _req.get(pth_url, timeout=120)
            resp.raise_for_status()
            pth_path = work / "model.pth"
            with open(pth_path, "wb") as f:
                f.write(resp.content)
        else:
            pth_path = decode_b64_file(pth_b64, work / "model.pth")
        log.info(f"Model file ready: {pth_path.stat().st_size / 1024:.1f} KB")

        # Index: download from URL or decode base64
        index_path = None
        if index_url:
            import requests as _req
            log.info(f"Downloading index from URL: {index_url[:80]}...")
            resp = _req.get(index_url, timeout=60)
            resp.raise_for_status()
            index_path = work / "model.index"
            with open(index_path, "wb") as f:
                f.write(resp.content)
        elif index_b64:
            index_path = decode_b64_file(index_b64, work / "model.index")
        if index_path:
            log.info(f"Index file ready: {index_path.stat().st_size / 1024:.1f} KB")

        # Decode input audio: download from URL or decode base64
        input_ext = Path(audio_filename).suffix.lower() or ".mp3"
        if audio_url:
            import requests as _req
            log.info(f"Downloading audio from URL: {audio_url[:80]}...")
            resp = _req.get(audio_url, timeout=120)
            resp.raise_for_status()
            raw_input = work / f"input{input_ext}"
            with open(raw_input, "wb") as f:
                f.write(resp.content)
        else:
            raw_input = decode_b64_file(audio_b64, work / f"input{input_ext}")
        log.info(f"Audio file ready: {raw_input.stat().st_size / 1024:.1f} KB")

        # Normalize to WAV for processing (keep STEREO for Demucs quality)
        input_stereo = work / "input_stereo.wav"
        run_ffmpeg([
            "-i", str(raw_input),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            str(input_stereo),  # preserve original channel count (stereo if source is stereo)
        ])
        # Also create mono version for direct RVC use (when skipping Demucs)
        input_wav = work / "input_normalized.wav"
        run_ffmpeg([
            "-i", str(raw_input),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            str(input_wav),
        ])

        # --- Step 2: Vocal separation with Demucs ---
        accomp_path = None
        if separate_vocals:
            runpod.serverless.progress_update(job, "Separating vocals (Demucs)... (2/4)")
            demucs_dir = ensure_dir(work / "demucs")
            # Use STEREO input for Demucs — stereo cues improve separation quality
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

        # --- Step 3: RVC voice conversion on vocals ---
        runpod.serverless.progress_update(job, "Running voice conversion (RVC)... (3/4)")

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

        # --- Step 4: Mix converted vocals + original accompaniment ---
        mixed_path = None
        if accomp_path and accomp_path.exists():
            runpod.serverless.progress_update(job, "Mixing vocals + accompaniment... (4/4)")
            mixed_path = work / f"mixed_output.{export_format}"
            try:
                _mix_audio(converted_vocals_path, accomp_path, mixed_path,
                           vocal_volume=vocal_volume, mr_volume=mr_volume)
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
                bucket_name = os.environ.get("BUCKET_NAME", "voice-studio")

                vocals_url = upload_file_to_bucket(
                    file_name=f"converted_{Path(audio_filename).stem}.{export_format}",
                    file_location=str(converted_vocals_path),
                    prefix=f"voice-studio/convert/{job_id}",
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
                        prefix=f"voice-studio/convert/{job_id}",
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
        cleanup_dir(work)
        cleanup_gpu()


def _rvc_infer(
    pth_path: Path,
    index_path: Optional[Path],
    input_audio: Path,
    output_path: Path,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.88,
    protect: float = 0.23,
    hop_length: int = 128,
    clean_audio: bool = True,
    clean_strength: float = 0.7,
    export_format: str = "wav",
    filter_radius: int = 3,
    rms_mix_rate: float = 0.1,
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
            split_audio=True,
            f0_autotune=False,
            f0_autotune_strength=1.0,
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
            split_audio=True,
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
        cpt = torch.load(str(pth_path), map_location="cpu")
        config = cpt.get("config", [])
        tgt_sr = config[-1] if config else 40000
        if not tgt_sr or tgt_sr <= 1:
            tgt_sr = 40000

        if "weight" in cpt and "emb_g.weight" in cpt["weight"]:
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
            input_audio,
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
            False,  # f0_autotune
            1.0,  # f0_autotune_strength
        )

        # Save output
        sf.write(str(output_path), audio_opt, tgt_sr, format="WAV")
        log.info(f"Inference completed via low-level pipeline (output: {output_path.stat().st_size / 1024:.1f} KB)")
        return
    except Exception as e:
        log.warning(f"Low-level pipeline failed: {e}", exc_info=True)
    finally:
        os.chdir(original_cwd)

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
        "--split_audio", "True",
        "--clean_audio", str(clean_audio),
        "--clean_strength", str(clean_strength),
        "--export_format", export_format.upper(),
        "--embedder_model", "contentvec",
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
        log.error(f"CLI failed with code {cli_result.returncode}")

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
                    import shutil
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
