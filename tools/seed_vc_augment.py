"""
Seed-VC 학습 데이터 증강 도구 (v54)

목적: 다른 가수의 고음/가성 노래를 장홍권 음색으로 변환하여
      RVC 학습 데이터에 추가 → 모델의 고음역 커버리지 확장

v54 변경사항:
  - seed-uvit-whisper-base 모델 사용 (노래 전용, 44.1kHz BigVGAN)
  - 파인튜닝 지원 (레퍼런스 1개로 100~500스텝 파인튜닝)
  - 데이터 증강 변형 생성 (time-stretch ±5%)
  - Rubber Band / WORLD 피치시프트 증강 추가

사용법:
  1. Seed-VC 설치: https://github.com/Plachtaa/seed-vc
  2. 고음 소스 곡 준비 (다른 가수의 C5-C6 가성 구간)
  3. 장홍권 레퍼런스 준비 (장이정-살다가테스트.mp3 권장)
  4. 이 스크립트 실행

워크플로우:
  1. (선택) Seed-VC 파인튜닝: 레퍼런스 음성으로 100~500스텝
  2. 소스 곡 → BS-Roformer 보컬 분리
  3. 분리된 보컬 → Seed-VC 제로샷/파인튜닝 변환 (장홍권 음색)
  4. 변환 결과 → time-stretch ±5% 변형 생성 (데이터 다양성)
  5. 결과 품질 필터링 (수동 청취 필수!)
  6. 통과된 파일 → preprocessed/ 에 추가
  7. 총 학습 데이터의 최대 20-30%까지만 증강 데이터 사용

주의사항:
  - Seed-VC는 GPL-3.0 라이선스 (도구 사용만, 출력물은 제한 없음)
  - 증강 데이터 비율 30% 초과 시 도메인 불일치로 품질 저하
  - 생성된 파일은 반드시 수동 청취 후 선별
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SEED_VC_DIR = Path(os.environ.get("SEED_VC_DIR", "~/seed-vc")).expanduser()
REFERENCE_AUDIO = Path("mp3record/장이정 - 살다가 한번쯤 테스트용.mp3")
OUTPUT_DIR = Path("augmented_vocals")
BS_ROFORMER_MODEL = Path("models/bs_roformer.ckpt")

# 고음 소스 곡 목록
HIGH_PITCH_SOURCES = [
    # "uploads/고음소스곡.mp3",
]

# Seed-VC 파라미터 (v54)
SEED_VC_MODEL = "seed-uvit-whisper-base"  # v54: 노래 전용 모델 (44.1kHz BigVGAN)
DIFFUSION_STEPS = 50      # 최대 품질
F0_CONDITION = True        # 멜로디 보존 (필수)
SEMI_TONE_SHIFT = 0        # 소스 곡 피치 유지
FINETUNE_STEPS = 200       # v54: 파인튜닝 스텝 (0=스킵, 100~500 권장)

# 데이터 증강 변형
TIME_STRETCH_VARIANTS = [0.95, 1.0, 1.05]  # v54: ±5% time-stretch


def check_dependencies():
    """Seed-VC 설치 확인"""
    if not SEED_VC_DIR.exists():
        log.error(f"Seed-VC not found at {SEED_VC_DIR}")
        log.info("Install: git clone https://github.com/Plachtaa/seed-vc.git")
        return False

    inference_py = SEED_VC_DIR / "inference.py"
    if not inference_py.exists():
        log.error(f"inference.py not found in {SEED_VC_DIR}")
        return False

    if not REFERENCE_AUDIO.exists():
        log.error(f"Reference audio not found: {REFERENCE_AUDIO}")
        return False

    return True


def separate_vocals(source_path: Path, output_dir: Path) -> Path:
    """BS-Roformer로 소스 곡에서 보컬 분리"""
    vocal_path = output_dir / f"{source_path.stem}_vocals.wav"
    if vocal_path.exists():
        log.info(f"  Cached: {vocal_path}")
        return vocal_path

    cmd = [
        sys.executable, "-m", "audio_separator",
        str(source_path),
        "--model_filename", "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "--output_dir", str(output_dir),
        "--output_format", "wav",
    ]
    log.info(f"  Separating vocals: {source_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning(f"  BS-Roformer failed ({e}), using source as-is")
        shutil.copy2(source_path, vocal_path)

    return vocal_path


def finetune_seed_vc(reference: Path, steps: int = 200):
    """Seed-VC 파인튜닝 (레퍼런스 음성 기반)"""
    if steps <= 0:
        log.info("Skipping Seed-VC fine-tuning (steps=0)")
        return True

    finetune_py = SEED_VC_DIR / "finetune.py"
    if not finetune_py.exists():
        log.warning("finetune.py not found, skipping fine-tuning")
        return False

    cmd = [
        sys.executable,
        str(finetune_py),
        "--reference", str(reference),
        "--steps", str(steps),
        "--model", SEED_VC_MODEL,
    ]
    log.info(f"Fine-tuning Seed-VC: {steps} steps with {reference.name}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        log.info(f"Fine-tuning completed: {result.stdout[-200:]}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Fine-tuning failed: {e.stderr[:200]}")
    except FileNotFoundError:
        log.warning("Python or finetune.py not found")
    return False


def seed_vc_convert(source_vocal: Path, reference: Path, output_path: Path):
    """Seed-VC로 음색 변환"""
    cmd = [
        sys.executable,
        str(SEED_VC_DIR / "inference.py"),
        "--source", str(source_vocal),
        "--target", str(reference),
        "--output", str(output_path),
        "--diffusion-steps", str(DIFFUSION_STEPS),
        "--semi-tone-shift", str(SEMI_TONE_SHIFT),
    ]
    if F0_CONDITION:
        cmd.extend(["--f0-condition", "True"])

    log.info(f"  Converting: {source_vocal.name} → {output_path.name}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            log.info(f"  SUCCESS: {output_path.name} ({size_mb:.1f} MB)")
            return True
    except subprocess.CalledProcessError as e:
        log.error(f"  Conversion error: {e.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log.error("  Timeout (600s)")

    return False


def create_time_stretch_variants(audio_path: Path, output_dir: Path) -> list:
    """FFmpeg로 time-stretch 변형 생성 (±5%)"""
    variants = []
    for rate in TIME_STRETCH_VARIANTS:
        if rate == 1.0:
            variants.append(audio_path)
            continue

        suffix = f"_ts{rate:.2f}"
        variant_path = output_dir / f"{audio_path.stem}{suffix}.wav"

        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-af", f"atempo={rate}",
            "-acodec", "pcm_s16le",
            str(variant_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            if variant_path.exists():
                variants.append(variant_path)
                log.info(f"  Time-stretch variant: {variant_path.name} ({rate:.2f}x)")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning(f"  Time-stretch {rate}x failed: {e}")

    return variants


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("=" * 60)
    log.info("Seed-VC Training Data Augmentation (v54)")
    log.info("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    if not HIGH_PITCH_SOURCES:
        log.info("\nNo high-pitch source songs configured!")
        log.info("Edit HIGH_PITCH_SOURCES in this script to add source songs.")
        log.info("\nRecommended sources (male tenor/falsetto, C5-C6):")
        log.info("  - Korean ballads with high falsetto")
        log.info("  - The Weeknd, Charlie Puth (English falsetto)")
        log.info("  - 플레이브, BTS 고음 파트")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)

    # v54: 파인튜닝 (선택적)
    if FINETUNE_STEPS > 0:
        finetune_seed_vc(REFERENCE_AUDIO, FINETUNE_STEPS)

    successful = 0
    total_variants = 0
    for source in HIGH_PITCH_SOURCES:
        source_path = Path(source)
        if not source_path.exists():
            log.info(f"SKIP: {source} (not found)")
            continue

        log.info(f"\nProcessing: {source_path.name}")

        # Step 1: Vocal separation
        vocal = separate_vocals(source_path, temp_dir)
        if not vocal.exists():
            log.info("  SKIP: Vocal separation failed")
            continue

        # Step 2: Seed-VC conversion
        output_name = f"augmented_{source_path.stem}.wav"
        output_path = OUTPUT_DIR / output_name

        if seed_vc_convert(vocal, REFERENCE_AUDIO, output_path):
            successful += 1

            # Step 3: v54 — Time-stretch 변형 생성
            variants = create_time_stretch_variants(output_path, OUTPUT_DIR)
            total_variants += len(variants)

    log.info("")
    log.info("=" * 60)
    log.info(f"Results: {successful}/{len(HIGH_PITCH_SOURCES)} converted, "
             f"{total_variants} total variants")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Listen to EVERY augmented file manually")
    log.info("  2. Delete any with artifacts/timbre drift")
    log.info("  3. Copy good files to preprocessed/ for training")
    log.info("  4. Keep augmented data <= 20-30% of total training data")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
