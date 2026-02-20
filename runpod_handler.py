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


def encode_file_b64(file_path: Path) -> str:
    """Read a file and return its base64 encoding."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


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

        # Compress segments to MP3 for transfer (RunPod response limit ~20MB)
        # WAV: 24 segments × 10s × 44.1kHz × 16bit ≈ 25+ MB base64 → exceeds limit
        # MP3 192kbps: same segments ≈ 5 MB base64 → fits safely
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
                "data_base64": encode_file_b64(encode_path),
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
                "data_base64": encode_file_b64(encode_path),
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
                "data_base64": encode_file_b64(encode_path),
                "duration_seconds": round(dur, 2),
            })

        log.info(f"Returning {len(segment_data)} segments ({total_duration:.1f}s), "
                 f"{len(accomp_data)} accompaniment, {len(vocal_data)} vocal files")
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
        audio_data, sr = sf.read(str(ap))
        total_samples = len(audio_data)
        total_duration = total_samples / sr

        if total_duration <= SEGMENT_MAX:
            # File is already short enough, keep as-is if >= SEGMENT_MIN
            if total_duration >= SEGMENT_MIN:
                out = output_dir / f"seg_{seg_idx:04d}.wav"
                sf.write(str(out), audio_data, samplerate=sr, subtype="PCM_16")
                all_segments.append(out)
                seg_idx += 1
            elif total_duration >= 2.0:
                # Keep very short clips too (will be less than 5s but still usable)
                out = output_dir / f"seg_{seg_idx:04d}.wav"
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

            out = output_dir / f"seg_{seg_idx:04d}.wav"
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
    sample_rate: int = job_input.get("sample_rate", 40000)  # 40k recommended for SVC
    epochs: int = job_input.get("epochs", 300)
    batch_size: int = job_input.get("batch_size", 0)  # 0 = auto-detect
    f0_method: str = job_input.get("f0_method", "rmvpe")
    embedder_model: str = job_input.get("embedder_model", "contentvec")
    save_every_epoch: int = job_input.get("save_every_epoch", 25)  # finer checkpoints for optimal model selection

    if not audio_files:
        raise ValueError("No audio_files provided for training")

    # Validate parameters
    if sample_rate not in (32000, 40000, 44100, 48000):
        sample_rate = 44100
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
    sr_map = {32000: "32k", 40000: "40k", 44100: "44k", 48000: "48k"}
    sr_label = sr_map.get(sample_rate, "44k")

    try:
        # --- Step 1: Decode audio files to dataset directory ---
        runpod.serverless.progress_update(job, f"Decoding {len(audio_files)} audio files... (1/6)")
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
        runpod.serverless.progress_update(job, "Preprocessing audio (slicing, resampling)... (2/6)")
        try:
            _rvc_preprocess(model_name, str(wav_dir), sample_rate, logs_dir)
        except Exception as e:
            raise RuntimeError(f"[2/6 전처리] {e}") from e

        # --- Step 3: Extract F0 ---
        runpod.serverless.progress_update(job, f"Extracting F0 ({f0_method})... (3/6)")
        try:
            _rvc_extract_f0(model_name, f0_method, sample_rate, logs_dir)
        except Exception as e:
            raise RuntimeError(f"[3/6 F0추출] {e}") from e

        # --- Step 4: Extract features (ContentVec embeddings) ---
        runpod.serverless.progress_update(job, f"Extracting features ({embedder_model})... (4/6)")
        try:
            _rvc_extract_features(model_name, embedder_model, sample_rate, logs_dir)
        except Exception as e:
            raise RuntimeError(f"[4/6 특징추출] {e}") from e

        # --- Step 5: Train ---
        runpod.serverless.progress_update(job, f"Training model ({epochs} epochs)... (5/6)")
        _rvc_train(
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
        runpod.serverless.progress_update(job, "Generating FAISS index... (6/6)")
        _rvc_create_index(model_name, logs_dir)

        # --- Collect output files ---
        pth_path = _find_best_pth(logs_dir, model_name)
        index_path = _find_index(logs_dir, model_name)

        if pth_path is None:
            raise RuntimeError(f"Training completed but no .pth model found in {logs_dir}")

        elapsed = time.time() - start_time

        result = {
            "model_name": model_name,
            "epochs_trained": epochs,
            "sample_rate": sample_rate,
            "training_time_seconds": round(elapsed, 1),
            "pth_data": encode_file_b64(pth_path),
            "pth_filename": pth_path.name,
        }

        if index_path is not None:
            result["index_data"] = encode_file_b64(index_path)
            result["index_filename"] = index_path.name
        else:
            result["index_data"] = ""
            result["index_filename"] = ""
            log.warning("No FAISS index file generated")

        return result

    finally:
        cleanup_dir(WORK_DIR / f"train_{job_id}")
        # Keep logs_dir as it contains the model — but clean after encoding
        cleanup_dir(logs_dir)
        cleanup_gpu()


def _rvc_preprocess(
    model_name: str, dataset_path: str, sample_rate: int, logs_dir: Path
) -> None:
    """
    RVC preprocessing: resample and organize audio for training.

    Self-contained implementation using FFmpeg — avoids Applio's import chain
    (core.py → launch_tensorboard → tensorboard, etc.) which frequently breaks.

    Creates the directory structure RVC training expects:
      - logs/{model_name}/0_gt_wavs/   → audio at target sample rate
      - logs/{model_name}/1_16k_wavs/  → audio at 16kHz (for F0/feature extraction)
    """
    gt_dir = ensure_dir(logs_dir / "0_gt_wavs")
    sr16k_dir = ensure_dir(logs_dir / "1_16k_wavs")

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

            padded = f"{idx:07d}"

            # Save at target sample rate (mono, 16-bit PCM, soxr HQ resampler)
            gt_path = gt_dir / f"{padded}.wav"
            run_ffmpeg([
                "-i", str(audio_file),
                "-af", f"aresample=resampler=soxr:precision=28:osr={sample_rate}",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(gt_path),
            ])

            # Save at 16kHz for F0 & feature extraction (soxr HQ resampler)
            sr16k_path = sr16k_dir / f"{padded}.wav"
            run_ffmpeg([
                "-i", str(audio_file),
                "-af", "aresample=resampler=soxr:precision=28:osr=16000",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(sr16k_path),
            ])

            idx += 1
        except Exception as e:
            log.warning(f"Failed to preprocess {audio_file.name}: {e}")
            continue

    if idx == 0:
        raise RuntimeError("전처리에 성공한 오디오 파일이 없습니다")

    log.info(f"RVC preprocess completed: {idx} files → {gt_dir} + {sr16k_dir}")


def _rvc_extract_f0(
    model_name: str, f0_method: str, sample_rate: int, logs_dir: Path
) -> None:
    """
    Extract F0 (fundamental frequency) using RMVPE or other methods.
    Tries multiple strategies with detailed error logging.
    """
    errors: list[str] = []

    # --- Strategy 1: Applio core API ---
    try:
        from core import run_extract_script
        run_extract_script(
            model_name=model_name,
            rvc_version="v2",
            f0_method=f0_method,
            hop_length=128,
            cpu_cores=os.cpu_count() or 4,
            sample_rate=str(sample_rate),
            embedder_model="contentvec",
            embedder_model_custom=None,
        )
        log.info(f"F0 extraction completed via core API (method={f0_method})")
        return
    except Exception as e:
        errors.append(f"core API: {type(e).__name__}: {e}")
        log.warning(f"F0 via core API failed: {e}")

    # --- Strategy 2: Direct module (rvc.train.extract) ---
    try:
        from rvc.train.extract.extract import run_pitch_extraction
        run_pitch_extraction(
            logs_dir=str(logs_dir),
            f0_method=f0_method,
        )
        log.info(f"F0 extraction completed via direct module (method={f0_method})")
        return
    except Exception as e:
        errors.append(f"direct module: {type(e).__name__}: {e}")
        log.warning(f"F0 via direct module failed: {e}")

    # --- Strategy 3: Legacy extract_f0_print ---
    try:
        from rvc.train.extract.extract_f0_print import extract_f0
        f0_dir = ensure_dir(logs_dir / "2a_f0")
        f0_nsf_dir = ensure_dir(logs_dir / "2b-f0nsf")
        wav_dir = logs_dir / "1_16k_wavs"
        if not wav_dir.exists():
            wav_dir = logs_dir / "0_gt_wavs"
        extract_f0(
            input_dir=str(wav_dir),
            f0_dir=str(f0_dir),
            f0nsf_dir=str(f0_nsf_dir),
            f0_method=f0_method,
        )
        log.info(f"F0 extraction completed via legacy module (method={f0_method})")
        return
    except Exception as e:
        errors.append(f"legacy module: {type(e).__name__}: {e}")
        log.warning(f"F0 via legacy module failed: {e}")

    # --- Strategy 4: CLI fallback ---
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(APPLIO_ROOT / "core.py"),
                "extract",
                "--model_name", model_name,
                "--f0_method", f0_method,
                "--sample_rate", str(sample_rate),
            ],
            cwd=str(APPLIO_ROOT),
            capture_output=True, text=True, timeout=7200, check=True,
        )
        log.info("F0 extraction completed via CLI")
        return
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "")[-500:]
        stdout_tail = (e.stdout or "")[-500:]
        errors.append(f"CLI exit code {e.returncode}: {stderr_tail or stdout_tail}")
        log.error(f"F0 CLI failed (exit {e.returncode}):\nSTDERR: {stderr_tail}\nSTDOUT: {stdout_tail}")
    except Exception as e:
        errors.append(f"CLI: {type(e).__name__}: {e}")
        log.error(f"F0 via CLI failed: {e}")

    raise RuntimeError(f"F0 추출 실패. 시도한 방법들: {'; '.join(errors)}")


def _rvc_extract_features(
    model_name: str, embedder_model: str, sample_rate: int, logs_dir: Path
) -> None:
    """
    Extract speaker embeddings using ContentVec (HuBERT-based) or other embedder.
    Note: If _rvc_extract_f0 used core API's run_extract_script, features may
    already be extracted. Check for existing features before running.
    """
    # Check if features were already extracted (by combined extract step)
    feature_dir = logs_dir / "3_feature768"
    if feature_dir.exists() and any(feature_dir.glob("*.npy")):
        log.info(f"Features already extracted ({len(list(feature_dir.glob('*.npy')))} files)")
        return

    errors: list[str] = []

    # --- Strategy 1: rvc.train.extract.extract ---
    try:
        from rvc.train.extract.extract import run_embedding_extraction
        run_embedding_extraction(
            logs_dir=str(logs_dir),
            embedder_model=embedder_model,
        )
        log.info(f"Feature extraction completed via run_embedding_extraction ({embedder_model})")
        return
    except Exception as e:
        errors.append(f"run_embedding_extraction: {type(e).__name__}: {e}")
        log.warning(f"Feature extraction via run_embedding_extraction failed: {e}")

    # --- Strategy 2: Legacy extract_feature_print ---
    try:
        from rvc.train.extract.extract_feature_print import extract_features
        wav_dir = logs_dir / "1_16k_wavs"
        if not wav_dir.exists():
            wav_dir = logs_dir / "0_gt_wavs"
        ensure_dir(feature_dir)
        extract_features(
            input_dir=str(wav_dir),
            output_dir=str(feature_dir),
            embedder_model=embedder_model,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        log.info(f"Feature extraction completed via legacy module ({embedder_model})")
        return
    except Exception as e:
        errors.append(f"extract_features: {type(e).__name__}: {e}")
        log.warning(f"Feature extraction via legacy module failed: {e}")

    # --- Strategy 3: CLI fallback ---
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(APPLIO_ROOT / "core.py"),
                "extract",
                "--model_name", model_name,
                "--embedder_model", embedder_model,
                "--sample_rate", str(sample_rate),
            ],
            cwd=str(APPLIO_ROOT),
            capture_output=True, text=True, timeout=7200, check=True,
        )
        log.info("Feature extraction completed via CLI")
        return
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "")[-500:]
        errors.append(f"CLI exit code {e.returncode}: {stderr_tail}")
        log.error(f"Feature CLI failed (exit {e.returncode}): {stderr_tail}")
    except Exception as e:
        errors.append(f"CLI: {type(e).__name__}: {e}")

    raise RuntimeError(f"특징 추출 실패. 시도한 방법들: {'; '.join(errors)}")


def _rvc_train(
    model_name: str,
    sample_rate: int,
    epochs: int,
    batch_size: int,
    save_every_epoch: int,
    logs_dir: Path,
    sr_label: str,
    job: dict,
) -> None:
    """
    Run RVC v2 training loop.
    Uses Applio's core.run_train_script or falls back to direct training module.
    """
    # Determine pretrained model paths
    pretrained_g = _find_pretrained("G", sr_label)
    pretrained_d = _find_pretrained("D", sr_label)

    try:
        from core import run_train_script

        run_train_script(
            model_name=model_name,
            rvc_version="v2",
            save_every_epoch=save_every_epoch,
            save_only_latest=False,
            save_every_weights=True,
            total_epoch=epochs,
            sample_rate=str(sample_rate),
            batch_size=batch_size,
            gpu="0",
            pitch_guidance=True,
            overtraining_detector=True,
            overtraining_threshold=50,
            pretrained=True,
            custom_pretrained=False,
            g_pretrained_path=str(pretrained_g) if pretrained_g else None,
            d_pretrained_path=str(pretrained_d) if pretrained_d else None,
        )
        log.info(f"Training completed via core API ({epochs} epochs)")

    except ImportError:
        try:
            # Direct module: construct training config and run
            from rvc.train.train import train as rvc_train_fn

            config = {
                "model_name": model_name,
                "sample_rate": sample_rate,
                "total_epoch": epochs,
                "batch_size": batch_size,
                "save_every_epoch": save_every_epoch,
                "gpu_ids": "0",
                "if_f0": True,       # pitch guidance
                "version": "v2",
                "logs_path": str(logs_dir),
                "pretrainedG": str(pretrained_g) if pretrained_g else "",
                "pretrainedD": str(pretrained_d) if pretrained_d else "",
            }
            rvc_train_fn(config)
            log.info(f"Training completed via direct train module ({epochs} epochs)")

        except ImportError:
            # CLI fallback
            cmd = [
                sys.executable,
                str(APPLIO_ROOT / "core.py"),
                "train",
                "--model_name", model_name,
                "--rvc_version", "v2",
                "--total_epoch", str(epochs),
                "--sample_rate", str(sample_rate),
                "--batch_size", str(batch_size),
                "--save_every_epoch", str(save_every_epoch),
                "--gpu", "0",
                "--pitch_guidance", "True",
                "--pretrained", "True",
            ]
            if pretrained_g:
                cmd.extend(["--g_pretrained_path", str(pretrained_g)])
            if pretrained_d:
                cmd.extend(["--d_pretrained_path", str(pretrained_d)])

            log.info(f"Starting training via CLI: {epochs} epochs")
            proc = subprocess.Popen(
                cmd,
                cwd=str(APPLIO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output and report progress
            epoch_count = 0
            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                log.info(f"[train] {line}")

                # Parse epoch progress from training output
                if "Epoch" in line or "epoch" in line:
                    try:
                        # Typical format: "Epoch 50/300" or "Training epoch 50"
                        for part in line.split():
                            if "/" in part:
                                current, total = part.split("/")
                                epoch_count = int(current)
                                break
                    except (ValueError, IndexError):
                        pass

                    if epoch_count > 0:
                        pct = min(95, int((epoch_count / epochs) * 100))
                        try:
                            runpod.serverless.progress_update(
                                job, f"Training epoch {epoch_count}/{epochs} ({pct}%)"
                            )
                        except Exception:
                            pass

            proc.wait(timeout=36000)  # 10-hour timeout
            if proc.returncode != 0:
                raise RuntimeError(
                    f"학습 프로세스 실패 (exit code {proc.returncode}). "
                    f"마지막 출력을 확인하세요."
                )
            log.info(f"Training completed via CLI ({epochs} epochs)")


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
    try:
        from core import run_index_script

        run_index_script(
            model_name=model_name,
            rvc_version="v2",
        )
        log.info("FAISS index created via core API")
    except ImportError:
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
    Find the best .pth model file from training output.
    Applio saves weights to logs/{model_name}/ and also
    to logs/{model_name}/weights/ or rvc/models/{model_name}.pth
    """
    candidates: list[tuple[Path, int]] = []

    # Search patterns
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
            # Extract epoch number from filename if present
            # Common patterns: model_name_e300_s1200.pth, model_name_300e.pth
            name = pth.stem
            epoch = 0
            try:
                for part in name.split("_"):
                    if part.startswith("e") and part[1:].isdigit():
                        epoch = int(part[1:])
                    elif part.endswith("e") and part[:-1].isdigit():
                        epoch = int(part[:-1])
                    elif part.isdigit():
                        epoch = max(epoch, int(part))
            except (ValueError, IndexError):
                pass
            candidates.append((pth, epoch))

    if not candidates:
        return None

    # Return highest epoch model (or most recent if no epoch info)
    candidates.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
    best = candidates[0][0]
    log.info(f"Best model found: {best}")
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
    index_b64: str = job_input.get("index_data", "")
    audio_b64: str = job_input.get("audio_data", "")
    audio_filename: str = job_input.get("audio_filename", "input.wav")
    pitch_shift: int = job_input.get("pitch_shift", 0)
    index_rate: float = job_input.get("index_rate", 0.75)
    f0_method: str = job_input.get("f0_method", "rmvpe")
    filter_radius: int = job_input.get("filter_radius", 3)
    rms_mix_rate: float = job_input.get("rms_mix_rate", 0.25)
    protect: float = job_input.get("protect", 0.33)
    hop_length: int = job_input.get("hop_length", 128)
    clean_audio: bool = job_input.get("clean_audio", False)
    clean_strength: float = job_input.get("clean_strength", 0.7)
    export_format: str = job_input.get("export_format", "wav")
    # SVC pipeline options
    separate_vocals: bool = job_input.get("separate_vocals", True)
    vocal_volume: float = job_input.get("vocal_volume", 1.0)
    mr_volume: float = job_input.get("mr_volume", 1.0)

    if not pth_b64:
        raise ValueError("No pth_data (model weights) provided for conversion")
    if not audio_b64:
        raise ValueError("No audio_data (input audio) provided for conversion")

    job_id = job.get("id", uuid.uuid4().hex[:12])
    work = ensure_dir(WORK_DIR / f"convert_{job_id}")
    start_time = time.time()

    try:
        # --- Step 1: Decode files ---
        runpod.serverless.progress_update(job, "Decoding model and audio files... (1/4)")

        pth_path = decode_b64_file(pth_b64, work / "model.pth")
        log.info(f"Model file decoded: {pth_path.stat().st_size / 1024:.1f} KB")

        index_path = None
        if index_b64:
            index_path = decode_b64_file(index_b64, work / "model.index")
            log.info(f"Index file decoded: {index_path.stat().st_size / 1024:.1f} KB")

        # Decode and normalize input audio to WAV
        input_ext = Path(audio_filename).suffix.lower()
        raw_input = decode_b64_file(audio_b64, work / f"input{input_ext}")

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
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            hop_length=hop_length,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format,
        )

        if not converted_vocals_path.exists():
            raise RuntimeError("Conversion produced no output file")

        # --- Step 4: Mix converted vocals + original accompaniment ---
        result = {
            "converted_audio": encode_file_b64(converted_vocals_path),
            "filename": f"converted_{Path(audio_filename).stem}.{export_format}",
            "parameters": {
                "pitch_shift": pitch_shift,
                "f0_method": f0_method,
                "index_rate": index_rate,
                "filter_radius": filter_radius,
                "rms_mix_rate": rms_mix_rate,
                "protect": protect,
            },
        }

        if accomp_path and accomp_path.exists():
            runpod.serverless.progress_update(job, "Mixing vocals + accompaniment... (4/4)")
            mixed_path = work / f"mixed_output.{export_format}"
            try:
                _mix_audio(converted_vocals_path, accomp_path, mixed_path,
                           vocal_volume=vocal_volume, mr_volume=mr_volume)
                if mixed_path.exists():
                    result["mixed_audio"] = encode_file_b64(mixed_path)
                    result["mixed_filename"] = f"mixed_{Path(audio_filename).stem}.{export_format}"
                    log.info("Mixed output created successfully")
            except Exception as e:
                log.error(f"Mixing failed (returning vocals only): {e}")

        elapsed = time.time() - start_time
        result["processing_time_seconds"] = round(elapsed, 2)
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
    index_rate: float = 0.75,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hop_length: int = 128,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    export_format: str = "wav",
) -> None:
    """
    Run RVC v2 inference using Applio's pipeline.
    Tries multiple import strategies for compatibility across Applio versions.
    """
    index_str = str(index_path) if index_path and index_path.exists() else ""

    # --- Strategy 1: Applio core API (>= v3.2.x) ---
    try:
        from core import run_infer_script

        run_infer_script(
            pitch=pitch_shift,
            filter_radius=filter_radius,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            hop_length=hop_length,
            f0_method=f0_method,
            input_path=str(input_audio),
            output_path=str(output_path),
            pth_path=str(pth_path),
            index_path=index_str,
            split_audio=True,    # split into chunks for more consistent quality
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format.upper(),
        )
        log.info("Inference completed via core.run_infer_script")
        return
    except ImportError:
        log.info("core.run_infer_script not available, trying alternative")
    except Exception as e:
        log.warning(f"core.run_infer_script failed: {e}, trying alternative")

    # --- Strategy 2: Direct VC pipeline import ---
    try:
        from rvc.infer.infer import VoiceConverter

        vc = VoiceConverter()
        vc.convert_audio(
            audio_input_path=str(input_audio),
            audio_output_path=str(output_path),
            model_path=str(pth_path),
            index_path=index_str,
            pitch=pitch_shift,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            hop_length=hop_length,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format.upper(),
        )
        log.info("Inference completed via VoiceConverter class")
        return
    except ImportError:
        log.info("rvc.infer.infer.VoiceConverter not available, trying alternative")
    except Exception as e:
        log.warning(f"VoiceConverter failed: {e}, trying alternative")

    # --- Strategy 3: Legacy VC pipeline ---
    try:
        from rvc.modules.vc.modules import VC

        vc = VC()
        vc.get_vc(str(pth_path))

        # VC.vc_single expects specific args depending on version
        result = vc.vc_single(
            sid=0,
            input_audio_path=str(input_audio),
            f0_up_key=pitch_shift,
            f0_method=f0_method,
            file_index=index_str,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=0,  # 0 = auto
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        # result is typically (message, (sample_rate, audio_array))
        if isinstance(result, tuple) and len(result) >= 2:
            sr_out, audio_out = result[1] if isinstance(result[1], tuple) else (44100, result[1])
        else:
            raise RuntimeError(f"Unexpected VC output format: {type(result)}")

        import soundfile as sf
        sf.write(str(output_path), audio_out, samplerate=sr_out)
        log.info("Inference completed via legacy VC module")
        return
    except ImportError:
        log.info("rvc.modules.vc.modules.VC not available, trying CLI fallback")
    except Exception as e:
        log.warning(f"Legacy VC failed: {e}, trying CLI fallback")

    # --- Strategy 4: CLI fallback ---
    cmd = [
        sys.executable,
        str(APPLIO_ROOT / "core.py"),
        "infer",
        "--pitch", str(pitch_shift),
        "--filter_radius", str(filter_radius),
        "--index_rate", str(index_rate),
        "--rms_mix_rate", str(rms_mix_rate),
        "--protect", str(protect),
        "--hop_length", str(hop_length),
        "--f0_method", f0_method,
        "--input_path", str(input_audio),
        "--output_path", str(output_path),
        "--pth_path", str(pth_path),
        "--index_path", index_str,
        "--export_format", export_format.upper(),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(APPLIO_ROOT),
        capture_output=True, text=True, timeout=600, check=True,
    )
    log.info("Inference completed via CLI fallback")


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
