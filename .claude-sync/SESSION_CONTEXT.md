# Session Context — 2026-04-07 (v43 세션)

## 마지막 작업: v43 후처리 + 시스템 총점검

### v42→v43 코드 변경
- 300Hz -1.5dB + 600Hz -1.0dB (가래/먹먹함 해결, 300-800Hz +3.6dB 과잉)
- 3kHz +2.0dB presence 부스트 (2-4kHz -4.4dB 발음 손실 복원)
- 고음곡 프리셋: pitch -5→-3 (F0 60Hz 바닥 도달 방지)
- post_reverb handler 기본값 0.05→0.0 (server와 일치)
- 콜드 스타트 타임아웃: convert 30→45분, preprocess 60→90분

### v43 후처리 체인 (현재)
```
highpass 70Hz → 300Hz -1.5dB → 600Hz -1.0dB →
3kHz +2.0dB (presence) → 6.5kHz -1.0dB/w=0.3 (디에서) →
(고음모드) → (리버브) → 2-pass loudnorm -14 LUFS (LRA=20)
```

### 핵심 파라미터 (v43)
- index_rate: 0.30, rms_mix_rate: 0.0, protect: 0.33, filter_radius: 3
- epochs: 200, batch: 8, SR: 40kHz, F0: RMVPE
- vocal_blend: 10% (프리셋), post_reverb: 0.0

### 곡별 최적 세팅
| 곡 | 프리셋 | pitch_shift |
|---|---|:---:|
| Breaking Through | 기본값 | 0 |
| 기다릴게 | 고음곡 전용 | -3 |
| comethru | 듀엣 | 0 |

### 테스트 상태
- 48/48 통과, ruff 클린
- git main, 최신 커밋

### 다음 할 일
1. v43 후처리로 3곡 재변환 (Docker 리빌드 후)
2. 결과 청취 후 미세 조정
3. 듀엣/화음곡: 리드/백킹 보컬 사전 분리 워크플로우 검토
