# Session Context — 2026-04-09 (v49.2 전면 개선 + 증강 도구)

## 마지막 작업: v49 시리즈 전면 개선

### v49.0 — 변환 파이프라인 (커밋 2f985c4)
- f0_autotune=True (strength=0.6): 가성 피치 안정화
- hop_length 64→128: 노이즈 추적→삑사리 방지
- filter_radius 2→3, protect 0.33→0.40
- 3kHz presence boost 제거 + 한/영 EQ 분리 (language 파라미터)
- 프리셋 12개 전면 재설계, UI 언어 드롭다운 추가

### v49.1 — 학습 최적화 (커밋 88d7c1e)
- epochs 200→150 (과적합→치찰음 방지)
- batch_size 4→6 (44.9분 데이터 최적)
- 학습 전처리 디에싱: 6kHz -2dB + 8.5kHz -1.5dB

### v49.2 — 증강 도구 (커밋 4474b55)
- tools/seed_vc_augment.py: Seed-VC 제로샷 고음역 확장
- tools/pitch_augment.py: WORLD 보코더 피치시프트 증강
- 향후 Amphion VevoSing (MIT) 지원 예정

### 70+ 음원 전수조사 결과
- 124개 파일 (mp3record 71 + uploads 53)
- 16kHz 대화: 33개 → 전부 제외
- _talkf_ 대화: 8개 → 전부 제외
- 외부 커버곡: 24개 (XIA OST x15, Lee Seung Gi x4 등) → 장홍권 아님, 제외
- 유니크 노래: 29개, 106.3분
- 추천 데이터셋: 이전과 동일 9개 (44.9분) + 증강 데이터 10-15분

### 제로샷 SVC 도구 비교 (12개 평가)
1. **Seed-VC** (GPL-3.0) — 최우선, BigVGAN 고음 특화
2. **Amphion VevoSing** (MIT) — MIT 필요 시, ICLR 2025
3. HQ-SVC — 소규모 데이터 특화 (AAAI 2026)
4. YingMusic-SVC — CC-BY-NC (비상업 제한)
5~12. GPT-SoVITS, CosyVoice, OpenVoice 등 — SVC 부적합

### 다음 할 일
1. **즉시**: v49 파라미터로 3곡 재변환 (기본값/한국어/영어 프리셋)
2. **에폭 비교**: 150 에폭으로 재학습 → 체크포인트 50/100/150 비교
3. **(선택) Seed-VC 증강**:
   - 고음 곡 3-5개 소스 준비 (다른 가수 C5-C6)
   - 장홍권 레퍼런스로 Seed-VC 변환
   - 품질 필터링 후 학습 데이터 추가 (≤20%)
4. **(선택) WORLD 피치 증강**: 깨끗한 보컬 +2/+4 반음

### 핵심 파라미터 (v49.2)
- index_rate: 0.45 (ko:0.55, en:0.35), rms: 0.0
- protect: 0.40, filter_radius: 3 (ko:4), hop_length: 128
- f0_autotune: True (0.6), language: auto/ko/en
- epochs: 150, batch: 6, SR: 40kHz
- 학습 전처리: 디에싱 6kHz -2dB + 8.5kHz -1.5dB
- 테스트: 48/48 통과
