# Session Context — 2026-04-10 (v49.9 전면 개편 완료 + Dockerfile 수정)

## 최근 세션 요약

### v49 시리즈 전면 개편 (2026-04-09~10, 19개 커밋)
18개+ 에이전트 풀가동, 하네스엔지니어링 기법으로 전처리→학습→변환→후처리 전체 파이프라인 대규모 개편.

### 해결한 문제 9가지
1. **고음/가성 끊김·삑사리** → f0_autotune + hop128 + filter3
2. **치찰음/기계음** → 3kHz boost 제거 + 디에싱 + epochs 150
3. **발음 부정확** → 한/영 EQ 분리 (language=ko/en/auto)
4. **화음 부자연스러움** → 리드/백킹 2-pass 분리 + 3트랙 믹싱 (mel_band_roformer_karaoke)
5. **감기 걸린 목소리(비음)** → 1.2kHz -1.0dB 공통 컷 + autotune 0.3
6. **목소리 뚝뚝 끊김** → Applio 청크 38→65초 + vp_threshold 0.01
7. **과적합→치찰음** → epochs 150, batch 8, 보수적 디에싱
8. **파라미터 불일치 버그** → 3파일(server.py/handler/html) 완전 정합성 검증
9. **Dockerfile 빌드 에러** → sed→Python re.sub 안전 패치

### 핵심 파라미터 (v49.9 최종)
| 파라미터 | 값 | 비고 |
|----------|-----|------|
| index_rate | 0.45 (ko:0.55, en:0.35) | 한/영 프리셋 분리 |
| protect | 0.40 | 0.33→0.40 |
| filter_radius | 3 (ko:4) | 2→3 |
| hop_length | 128 | 64→128 |
| f0_autotune | True (strength=0.3) | 0.6→0.3 비음 완화 |
| epochs | 150 | 250→150 과적합 방지 |
| batch_size | 8 | AI Hub >30분=8 |
| language | auto/ko/en | 한/영 EQ 분리 (NEW) |
| vocal_blend | 0.0 | 더블링 방지 |
| post_reverb | 0.0 | 리버브 비활성 |
| 보컬 분리 | BS-Roformer → Demucs 폴백 | SDR 12.9 |
| 리드/백킹 분리 | mel_band_roformer_karaoke | SDR 10.20 (NEW) |
| 후처리 EQ | 1.2kHz -1.0 + 8kHz -0.8 (공통) | v49.8 |
| 후처리 EQ (en) | + 300Hz -0.3 + 600Hz -0.3 + 5kHz -0.8 | |
| 학습 디에싱 | 8.5kHz -1.0dB | 치찰음 학습 방지 |
| Applio 청크 | x_center=60, x_max=65 | 끊김 감소 |

### 증강 도구 (tools/)
- `tools/seed_vc_augment.py`: Seed-VC 제로샷 고음역 학습 데이터 생성
- `tools/pitch_augment.py`: Rubber Band/WORLD 포먼트 보존 피치시프트

### 학습 데이터 (9개 소수정예, 44.9분)
- S-tier: 장이정 학습용(20.0m), 살다가테스트(3.4m), 히스토리(1.1m)
- A-tier: 156a072e(4.2m), 6dea4ebc(4.3m), a2eebb1e(4.0m), 살다가한번쯤(3.9m), 5a8a4c9a(3.0m), 90d37ee7(2.5m)
- 70+ 파일 전수조사 → 16kHz 대화 33개 + _talkf_ 8개 + 외부 커버 24개 제외

### 향후 로드맵
1. Docker 재빌드 후 3곡 재변환 테스트
2. My Voice v42 재학습: epochs 150, batch 8
3. Spin V2 embedder A/B 테스트 (발음 개선)
4. Mel-RoFormer 보컬 분리 업그레이드 (+0.5dB SDR)
5. Seed-VC 고음 증강 (C5-C6 커버리지)
6. PyTorch 2.6.0+ CVE 대응

### Dockerfile 수정 이력 (주의!)
- v49.8: Applio config.py 패치 추가 (청크 크기 확대)
- 1차 시도: sed → SyntaxError (`65self.device_config()`)
- 2차 시도: Python if/else → Docker 빌드 에러 (들여쓰기 깨짐)
- 3차 최종: shell if + Python 세미콜론 단일라인 → 성공
