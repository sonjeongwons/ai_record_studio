"""구간별 세밀 분석 — AI 아티팩트 집중 구간 찾기"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Dict, Any, List

# --- 분석 설정 및 임계값 ---
SEGMENT_DURATION_SEC = 10  # 분석 구간 길이 (초)
MIN_SEGMENT_DURATION_SR = 1 # 분석할 최소 길이 (초)

# 문제 판정 임계값
BRIGHTNESS_DELTA_THRESHOLD = 400  # 음색 변화 임계값 (Hz)
PITCH_JUMP_COUNT_THRESHOLD = 3      # 피치 점프 임계 횟수
PITCH_JUMP_HZ_THRESHOLD = 100       # 피치 점프로 간주할 음높이 차이 (Hz)
FLATNESS_THRESHOLD = 0.005          # 노이즈성(평탄도) 임계값
ZCR_DELTA_THRESHOLD = 0.02          # 잡음(ZCR) 변화 임계값
VOLUME_UP_DB_THRESHOLD = 3          # 볼륨 증가 임계값 (dB)
VOLUME_DOWN_DB_THRESHOLD = -5       # 볼륨 감소 임계값 (dB)

HIGH_PITCH_THRESHOLD_HZ = 300       # 고음 구간으로 판단할 기준 주파수 (Hz)
ARTIFACT_FLATNESS_DIFF_THRESHOLD = 0.005 # 기계음 의심 구간 평탄도 차이 임계값
ARTIFACT_MIN_DURATION_SEC = 0.3     # 기계음 의심 구간 최소 길이 (초)

def _calculate_features(segment: np.ndarray, sr: int) -> Dict[str, Any]:
    """주요 오디오 특성을 계산합니다."""
    features = {}
    features['rms'] = librosa.amplitude_to_db(np.sqrt(np.mean(segment**2)) + 1e-10)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)[0])
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=segment)[0])
    return features

def _analyze_pitch(segment: np.ndarray, sr: int) -> int:
    """세그먼트의 피치 점프 횟수를 계산합니다."""
    f0, _, _ = librosa.pyin(segment, fmin=65, fmax=2000, sr=sr, frame_length=2048) # type: ignore
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 1:
        f0_diff = np.abs(np.diff(f0_valid))
        return int(np.sum(f0_diff > PITCH_JUMP_HZ_THRESHOLD))
    return 0

def analyze_segments(converted_path: str, original_path: str):
    """10초 단위로 원곡 vs 변환곡 비교 — 문제 구간 특정"""

    print(f"\n{'='*80}")
    print(f"  구간별 비교 분석: 원곡 vs v6 변환곡")
    print(f"{'='*80}\n")

    y_conv, sr = librosa.load(converted_path, sr=None, mono=True) # type: ignore
    y_orig, _ = librosa.load(original_path, sr=sr, mono=True) # type: ignore

    # 길이 맞추기
    min_len = min(len(y_conv), len(y_orig))
    y_conv = y_conv[:min_len]
    y_orig = y_orig[:min_len]

    duration = min_len / sr

    print(f"{'구간':>12s} | {'원곡RMS':>8s} {'변환RMS':>8s} {'ΔRMS':>6s} | "
          f"{'원곡밝기':>8s} {'변환밝기':>8s} {'Δ밝기':>7s} | "
          f"{'원곡ZCR':>8s} {'변환ZCR':>8s} {'ΔZCR':>7s} | "
          f"{'피치점프':>8s} {'평탄도':>7s} | {'판정':>6s}")
    print("-" * 130)

    problem_segments = []

    for start_sec in np.arange(0, duration, SEGMENT_DURATION_SEC):
        end_sec = min(start_sec + SEGMENT_DURATION_SEC, duration)
        start_s = int(start_sec * sr) # type: ignore
        end_s = int(end_sec * sr)

        seg_conv = y_conv[start_s:end_s]
        seg_orig = y_orig[start_s:end_s]

        if len(seg_conv) < sr:  # 1초 미만 스킵
            continue

        feat_conv = _calculate_features(seg_conv, sr)
        feat_orig = _calculate_features(seg_orig, sr)
        pitch_jumps = _analyze_pitch(seg_conv, sr)

        # 피치 분석
        f0: np.ndarray
        f0, _, _ = librosa.pyin(seg_conv, fmin=65, fmax=2000, sr=sr, frame_length=2048) # type: ignore
        f0_valid = f0[~np.isnan(f0)]
        pitch_jumps = 0
        if len(f0_valid) > 1:
            f0_diff = np.abs(np.diff(f0_valid))
            pitch_jumps = int(np.sum(f0_diff > PITCH_JUMP_HZ_THRESHOLD))

        # 문제 판정
        issues = []
        delta_brightness = feat_conv['spectral_centroid'] - feat_orig['spectral_centroid']
        delta_zcr = feat_conv['zcr'] - feat_orig['zcr']
        delta_rms = feat_conv['rms'] - feat_orig['rms']

        if abs(delta_brightness) > BRIGHTNESS_DELTA_THRESHOLD:
            issues.append("음색")
        if pitch_jumps >= PITCH_JUMP_COUNT_THRESHOLD:
            issues.append("피치")
        if feat_conv['spectral_flatness'] > FLATNESS_THRESHOLD:
            issues.append("노이즈")
        if delta_zcr > ZCR_DELTA_THRESHOLD:
            issues.append("잡음")
        if delta_rms > VOLUME_UP_DB_THRESHOLD:
            issues.append("볼륨↑")
        if delta_rms < VOLUME_DOWN_DB_THRESHOLD:
            issues.append("볼륨↓")

        verdict = ",".join(issues) if issues else "OK"

        time_str = f"{int(start_sec//60)}:{start_sec%60:04.1f}-{int(end_sec//60)}:{end_sec%60:04.1f}"

        print(f"{time_str:>12s} | {feat_orig['rms']:>7.1f}dB {feat_conv['rms']:>7.1f}dB {delta_rms:>+5.1f} | "
              f"{feat_orig['spectral_centroid']:>7.0f}Hz {feat_conv['spectral_centroid']:>7.0f}Hz {delta_brightness:>+6.0f} | "
              f"{feat_orig['zcr']:>8.4f} {feat_conv['zcr']:>8.4f} {delta_zcr:>+6.4f} | "
              f"{pitch_jumps:>6d}회 {feat_conv['spectral_flatness']:>6.4f} | {verdict}")

        if issues:
            problem_segments.append({
                'time': time_str,
                'start': start_sec,
                'end': end_sec,
                'issues': issues,
                'pitch_jumps': pitch_jumps,
                'delta_brightness': delta_brightness,
                'flatness': feat_conv['spectral_flatness'],
            })

    # ── 문제 구간 요약 ──
    print(f"\n{'='*80}")
    print(f"  문제 구간 요약 ({len(problem_segments)}개)")
    print(f"{'='*80}")

    if not problem_segments:
        print("  모든 구간 정상!")
    else:
        # 이슈 유형별 분류
        issue_types = {}
        for seg in problem_segments:
            for iss in seg['issues']:
                if iss not in issue_types:
                    issue_types[iss] = []
                issue_types[iss].append(seg)

        for iss_type, segs in sorted(issue_types.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{iss_type}] — {len(segs)}개 구간:")
            for s in segs:
                detail = ""
                if "피치" in s['issues']:
                    detail += f" 피치점프={s['pitch_jumps']}회"
                if "음색" in s['issues']:
                    detail += f" Δ밝기={s['delta_brightness']:+.0f}Hz"
                if "노이즈" in s['issues']:
                    detail += f" 평탄도={s['flatness']:.4f}"
                print(f"    {s['time']}{detail}")

    # ── 고음 구간 분석 (AI 아티팩트 집중) ──
    print(f"\n{'='*80}")
    print(f"  고음 구간 상세 분석 (F0 > 300Hz)")
    print(f"{'='*80}")

    f0_full, _, _ = librosa.pyin(y_conv, fmin=65, fmax=2000, sr=sr, frame_length=2048) # type: ignore
    hop_time = 512 / sr # type: ignore

    high_regions = []
    in_high = False
    start_t = 0

    for i, freq in enumerate(f0_full): # type: ignore
        t = i * hop_time
        if not np.isnan(freq) and freq > 300:
            if not in_high:
                start_t = t
                in_high = True
        else:
            if in_high:
                if t - start_t > 0.5:  # 0.5초 이상만
                    high_regions.append((start_t, t))
                in_high = False

    if high_regions:
        print(f"  고음 구간 {len(high_regions)}개 발견:")
        for st, en in high_regions:
            # 해당 구간의 세부 분석
            ss, es = int(st * sr), int(en * sr) # type: ignore
            seg = y_conv[ss:es]
            if len(seg) < 512:
                continue

            sc = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)[0])
            sf = np.mean(librosa.feature.spectral_flatness(y=seg)[0])

            f0_seg = f0_full[int(st/hop_time):int(en/hop_time)] # type: ignore
            f0_v = f0_seg[~np.isnan(f0_seg)]
            f0_max = np.max(f0_v) if len(f0_v) > 0 else 0
            f0_std = np.std(f0_v) if len(f0_v) > 1 else 0

            note = librosa.hz_to_note(f0_max) if f0_max > 0 else "?"
            stability = "안정" if f0_std < 20 else "불안정" if f0_std < 50 else "매우불안정"
            noise_flag = " [노이즈!]" if sf > 0.003 else ""

            print(f"    {st:.1f}~{en:.1f}초 ({en-st:.1f}s) | "
                  f"최고음={f0_max:.0f}Hz ({note}) | "
                  f"안정성={stability} (σ={f0_std:.1f}) | "
                  f"밝기={sc:.0f}Hz | 평탄도={sf:.4f}{noise_flag}")
    else:
        print("  300Hz 이상 고음 구간 없음")

    # ── 스펙트럴 평탄도 이상 구간 (기계음 후보) ──
    print(f"\n{'='*80}")
    print(f"  기계음/아티팩트 의심 구간 (스펙트럴 평탄도 급상승)")
    print(f"{'='*80}")

    sf_full = librosa.feature.spectral_flatness(y=y_conv, n_fft=2048, hop_length=512)[0]
    sf_orig_full = librosa.feature.spectral_flatness(y=y_orig, n_fft=2048, hop_length=512)[0]

    min_sf_len = min(len(sf_full), len(sf_orig_full))
    sf_diff = sf_full[:min_sf_len] - sf_orig_full[:min_sf_len]

    # 평탄도가 원곡 대비 급증하는 구간 찾기
    in_artifact = False
    artifact_regions = []

    for i in range(min_sf_len):
        t = i * 512 / sr # type: ignore
        if sf_diff[i] > ARTIFACT_FLATNESS_DIFF_THRESHOLD and sf_full[i] > 0.003:
            if not in_artifact:
                art_start = t
                in_artifact = True
        else:
            if in_artifact:
                if t - art_start > ARTIFACT_MIN_DURATION_SEC:
                    artifact_regions.append((art_start, t, np.max(sf_diff[int(art_start*sr/512):i])))
                in_artifact = False

    if artifact_regions:
        print(f"  {len(artifact_regions)}개 아티팩트 의심 구간:")
        for st, en, peak_diff in artifact_regions:
            print(f"    {st:.1f}~{en:.1f}초 ({en-st:.1f}s) — 평탄도 차이 최대 {peak_diff:.4f}")
    else:
        print("  뚜렷한 기계음 아티팩트 구간 없음 (원곡 대비 평탄도 정상)")

    print()

if __name__ == "__main__":
    converted = sys.argv[1] if len(sys.argv) > 1 else r"c:\ai_record_studio\mixed_플레이브 - 기다릴게4.wav"
    original = sys.argv[2] if len(sys.argv) > 2 else r"c:\ai_record_studio\플레이브 - 기다릴게.mp3"
    analyze_segments(converted, original)
