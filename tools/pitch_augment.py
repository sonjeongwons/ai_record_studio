"""
피치 증강 도구 (v49.3) — Rubber Band + WORLD 보코더

목적: 기존 학습 데이터를 피치시프트하여 음역 확장 샘플 생성
      Rubber Band --formant: 포먼트 보존하면서 피치 변환 (커뮤니티 검증)
      WORLD 보코더: Rubber Band 없을 때 대안 (asetrate와 다름!)

사용법:
  pip install pyworld soundfile numpy
  python tools/pitch_augment.py

원리:
  WORLD = 스펙트럼 엔벨로프(포먼트) + F0 + 비주기성 분해
  → F0만 시프트하면 포먼트 유지 + 피치만 변경
  → asetrate(재생속도 변경)와 근본적으로 다른 접근

주의:
  - +2/-2 반음만 안전 (커뮤니티 합의). +5 이상은 품질 저하 심함
  - 증강 데이터는 총 학습 데이터의 15-20%까지만 권장
  - 반드시 수동 청취 후 선별
  - 시간 늘리기(time stretch) 사용 금지 — 비브라토/자음 패턴 파괴
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def pitch_shift_rubberband(input_path: str, output_path: str, semitones: float) -> bool:
    """Rubber Band으로 포먼트 보존 피치시프트 (커뮤니티 최우선 권장)"""
    try:
        cmd = [
            "rubberband",
            "--formant",           # 포먼트 보존 (핵심!)
            "--pitch", str(semitones),
            "--crisp", "5",        # 보컬 최적 (기본값)
            input_path,
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except FileNotFoundError:
        print("  WARNING: rubberband CLI not found, falling back to WORLD vocoder")
        print("  Install: pip install pyrubberband  OR  apt install rubberband-cli")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: rubberband failed: {e}")
        return False


def pitch_shift_world(input_path: str, output_path: str, semitones: float):
    """WORLD 보코더로 포먼트 보존 피치 시프트"""
    try:
        import pyworld as pw
        import soundfile as sf
    except ImportError:
        print("pip install pyworld soundfile")
        return False

    # Load audio
    audio, sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    audio = audio.astype(np.float64)

    # WORLD analysis
    f0, timeaxis = pw.harvest(audio, sr, frame_period=5.0)
    sp = pw.cheaptrick(audio, f0, timeaxis, sr)
    ap = pw.d4c(audio, f0, timeaxis, sr)

    # Shift F0 (preserve formants via spectral envelope)
    ratio = 2 ** (semitones / 12.0)
    f0_shifted = f0.copy()
    f0_shifted[f0 > 0] *= ratio  # only shift voiced frames

    # Clamp to reasonable range
    f0_shifted = np.clip(f0_shifted, 0, 1200)

    # Synthesize
    audio_shifted = pw.synthesize(f0_shifted, sp, ap, sr)

    # Normalize
    max_val = np.max(np.abs(audio_shifted))
    if max_val > 0:
        audio_shifted = audio_shifted / max_val * 0.95

    sf.write(output_path, audio_shifted.astype(np.float32), sr)
    return True


def main():
    print("WORLD Vocoder Pitch Augmentation (v49)")
    print("=" * 50)

    # Source files (clean vocals from training data)
    source_dir = Path("mp3record")
    output_dir = Path("augmented_pitch")
    output_dir.mkdir(exist_ok=True)

    # Best clean vocal sources for augmentation
    clean_sources = [
        "장이정 - 살다가 한번쯤 테스트용.mp3",  # S-tier, 48kHz
        "히스토리_tomorrow_장이정파트 (1).mp3",   # S-tier, 48kHz
    ]

    # Pitch shifts: +2/-2만 안전 (커뮤니티 합의, SPA-SVC 논문)
    shifts = [+2, -2]  # +2 = 1음 올림 (고음 확장), -2 = 1음 내림 (저음 확장)

    total = 0
    for src_name in clean_sources:
        src_path = source_dir / src_name
        if not src_path.exists():
            print(f"SKIP: {src_name} (not found)")
            continue

        for shift in shifts:
            out_name = f"aug_{'+' if shift > 0 else ''}{shift}st_{src_path.stem}.wav"
            out_path = output_dir / out_name

            print(f"  {src_name} → {'+' if shift > 0 else ''}{shift} semitones")
            # Rubber Band 우선, 실패 시 WORLD 보코더 폴백
            success = pitch_shift_rubberband(str(src_path), str(out_path), shift)
            if not success:
                success = pitch_shift_world(str(src_path), str(out_path), shift)
            if success:
                size_mb = out_path.stat().st_size / 1024 / 1024
                print(f"    OK: {out_path.name} ({size_mb:.1f} MB)")
                total += 1
            else:
                print(f"    FAILED")

    print()
    print(f"Generated {total} augmented files in {output_dir}/")
    print("NEXT: Listen to each file, then copy good ones to preprocessed/")
    print("LIMIT: Max 10-15% of total training data should be augmented")


if __name__ == "__main__":
    main()
