"""
Seed-VC 학습 데이터 증강 도구 (v49)

목적: 다른 가수의 고음/가성 노래를 장홍권 음색으로 변환하여
      RVC 학습 데이터에 추가 → 모델의 고음역 커버리지 확장

사용법:
  1. Seed-VC 설치: https://github.com/Plachtaa/seed-vc
  2. 고음 소스 곡 준비 (다른 가수의 C5-C6 가성 구간)
  3. 장홍권 레퍼런스 준비 (장이정-살다가테스트.mp3 권장)
  4. 이 스크립트 실행

워크플로우:
  1. 소스 곡 → BS-Roformer 보컬 분리
  2. 분리된 보컬 → Seed-VC 제로샷 변환 (장홍권 음색)
  3. 결과 품질 필터링 (수동 청취 필수!)
  4. 통과된 파일 → preprocessed/ 에 추가
  5. 총 학습 데이터의 최대 20-30%까지만 증강 데이터 사용

주의사항:
  - Seed-VC는 GPL-3.0 라이선스 (도구 사용만, 출력물은 제한 없음)
  - 증강 데이터 비율 30% 초과 시 도메인 불일치로 품질 저하
  - 생성된 파일은 반드시 수동 청취 후 선별
  - 고음역 변환 품질이 낮으면 사용하지 않는 것이 나음
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SEED_VC_DIR = Path(os.environ.get("SEED_VC_DIR", "~/seed-vc")).expanduser()
REFERENCE_AUDIO = Path("mp3record/장이정 - 살다가 한번쯤 테스트용.mp3")  # 깨끗한 보컬
OUTPUT_DIR = Path("augmented_vocals")
BS_ROFORMER_MODEL = Path("models/bs_roformer.ckpt")  # 보컬 분리 모델

# 고음 소스 곡 목록 (다른 가수의 고음/가성 곡)
# 이 곡들의 보컬을 장홍권 음색으로 변환
HIGH_PITCH_SOURCES = [
    # 예시: uploads/ 에서 외부 커버곡 활용
    # "uploads/input_dd84dad4_BUZZ_-_Monologue_(mp3.pm).mp3",
    # 또는 별도 준비한 고음 곡
]

# Seed-VC 파라미터
DIFFUSION_STEPS = 50      # 최대 품질 (속도 희생)
F0_CONDITION = True        # 멜로디 보존
SEMI_TONE_SHIFT = 0        # 소스 곡 피치 유지 (이미 고음)


def check_dependencies():
    """Seed-VC 설치 확인"""
    if not SEED_VC_DIR.exists():
        print(f"ERROR: Seed-VC not found at {SEED_VC_DIR}")
        print("Install: git clone https://github.com/Plachtaa/seed-vc.git")
        print("Then: cd seed-vc && pip install -r requirements.txt")
        return False

    inference_py = SEED_VC_DIR / "inference.py"
    if not inference_py.exists():
        print(f"ERROR: inference.py not found in {SEED_VC_DIR}")
        return False

    if not REFERENCE_AUDIO.exists():
        print(f"ERROR: Reference audio not found: {REFERENCE_AUDIO}")
        print("Use the cleanest vocal recording (48kHz, 320kbps)")
        return False

    return True


def separate_vocals(source_path: Path, output_dir: Path) -> Path:
    """BS-Roformer로 소스 곡에서 보컬 분리"""
    vocal_path = output_dir / f"{source_path.stem}_vocals.wav"
    if vocal_path.exists():
        print(f"  Cached: {vocal_path}")
        return vocal_path

    # audio-separator 사용 (BS-Roformer)
    cmd = [
        sys.executable, "-m", "audio_separator",
        str(source_path),
        "--model_filename", "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "--output_dir", str(output_dir),
        "--output_format", "wav",
    ]
    print(f"  Separating vocals: {source_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: BS-Roformer failed, trying ffmpeg fallback")
        # Fallback: 그냥 원본 사용 (이미 보컬만 있다면)
        shutil.copy2(source_path, vocal_path)

    return vocal_path


def seed_vc_convert(source_vocal: Path, reference: Path, output_path: Path):
    """Seed-VC로 음색 변환 (소스 보컬 → 장홍권 음색)"""
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

    print(f"  Converting: {source_vocal.name} → {output_path.name}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  SUCCESS: {output_path.name} ({size_mb:.1f} MB)")
            return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout (600s)")

    return False


def main():
    print("=" * 60)
    print("Seed-VC Training Data Augmentation (v49)")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    if not HIGH_PITCH_SOURCES:
        print("WARNING: No high-pitch source songs configured!")
        print("Edit HIGH_PITCH_SOURCES in this script to add source songs.")
        print()
        print("Recommended sources (male tenor/falsetto, C5-C6):")
        print("  - Korean ballads with high falsetto")
        print("  - The Weeknd, Charlie Puth (English falsetto)")
        print("  - 플레이브, BTS 고음 파트")
        print()
        print("Steps:")
        print("  1. Download clean vocal-only tracks (or full songs)")
        print("  2. Add paths to HIGH_PITCH_SOURCES list")
        print("  3. Re-run this script")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)

    successful = 0
    for source in HIGH_PITCH_SOURCES:
        source_path = Path(source)
        if not source_path.exists():
            print(f"SKIP: {source} (not found)")
            continue

        print(f"\nProcessing: {source_path.name}")

        # Step 1: Vocal separation
        vocal = separate_vocals(source_path, temp_dir)
        if not vocal.exists():
            print(f"  SKIP: Vocal separation failed")
            continue

        # Step 2: Seed-VC conversion
        output_name = f"augmented_{source_path.stem}.wav"
        output_path = OUTPUT_DIR / output_name

        if seed_vc_convert(vocal, REFERENCE_AUDIO, output_path):
            successful += 1

    print()
    print("=" * 60)
    print(f"Results: {successful}/{len(HIGH_PITCH_SOURCES)} converted")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("NEXT STEPS:")
    print("  1. Listen to EVERY augmented file manually")
    print("  2. Delete any with artifacts/timbre drift")
    print("  3. Copy good files to preprocessed/ for training")
    print("  4. Keep augmented data ≤ 20-30% of total training data")
    print("=" * 60)


if __name__ == "__main__":
    main()
