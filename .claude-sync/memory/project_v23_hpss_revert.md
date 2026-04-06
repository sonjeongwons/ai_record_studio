---
name: v23 HPSS 리버트 및 품질 개선 방향
description: v22 HPSS 화음필터 파괴 확인, v23 리버트, 향후 개선 로드맵
type: project
---

## v22 HPSS 화음필터 실패 (2026-03-23)

v22의 HPSS(Harmonic-Percussive Source Separation)가 보컬 품질을 치명적으로 파괴.

**Why:** HPSS는 harmonic/percussive 분리이지, 리드/배경 보컬 분리가 아님.
보컬의 자음/숨소리가 percussive → HPSS harmonic 추출 시 모두 제거됨.

**실측 피해 (v17→v18):**
- ZCR -17% (자음/발음 손실)
- Presence(2-4kHz) -2.0dB
- Air(8-16kHz) -2.3dB
- 결과: 웅얼웅얼 발음 + 기계음 심화 → 노래 인식 불가

## v23 수정 (커밋 9e402ee + 29df6b0)

1. 모든 프리셋 harmonyFilter=0 (chorus/chorus_v2/duet 포함)
2. HPSS 함수 유지하되 기본 비활성 + 경고 로그
3. index_rate 0.35→0.50 상향 (커뮤니티 권장 0.50-0.75)

## 핵심 분석 결과

- v15 ≈ v17 (모든 지표 0.2dB 이내) — 파라미터 튜닝 한계 도달
- v1(첫 변환)의 SF가 v17의 3.3배 → HiFi-GAN 보코더의 멸균적 출력이 근본 문제
- 오버트레이닝이 기계음 #1 원인 (RVC 커뮤니티 합의)

## 향후 개선 로드맵

1. **모델 재학습 (v31)** — epoch 200-300으로 단축 (오버트레이닝 해소)
2. **리드 보컬 분리** — `audio-separator` 패키지 (UVR5 MDX-Net 모델)
   - Stage 1: Demucs → 악기에서 보컬 분리
   - Stage 2: MDX-Net (Kim Vocal 2) → 리드/배경 보컬 분리
3. **원본 보컬 블렌딩** — RVC 출력에 원본 보컬 5-10% 혼합 → 숨결감 복원

**How to apply:** v23으로 재변환하면 v17 수준 복원 + index_rate↑. 근본 개선은 v31 재학습 필요.
