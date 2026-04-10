# Session Context — 2026-04-10 (v49.9 최종 정합성 검증 완료)

## v49 시리즈 완료 현황 (16개 커밋)

### v49.0 — 변환 파이프라인 전면 개선
- f0_autotune=True (strength=0.3): 가성 피치 안정화
- hop_length 64→128, filter_radius 2→3, protect 0.33→0.40
- 한/영 EQ 분리 (language 파라미터), 프리셋 12개, UI 언어 드롭다운

### v49.1~v49.4 — 학습 최적화
- epochs 200→150 (과적합→치찰음 방지)
- batch_size →8 (AI Hub: >30분=8)
- 보수적 디에싱 8.5kHz -1.0dB

### v49.5~v49.6 — 화음 처리
- 리드/백킹 2-pass 분리 (mel_band_roformer_karaoke SDR 10.20)
- 리드만 RVC 변환, 백킹 원본 유지 (0.65볼륨 + 4kHz 롤오프)
- Dockerfile: karaoke 모델 사전 캐시

### v49.7 — 파이프라인 감사 버그 5건 수정
- index_rate/filter_radius/batch 기본값 불일치 해소
- f0_autotune configurable (job_input에서 받음)

### v49.8 — 비음/끊김 해결
- 1.2kHz -1.0dB 공통 EQ (HiFi-GAN 비음 공명)
- f0_autotune_strength 0.6→0.3 (이중 스무딩 완화)
- Applio 청크 38→65초 (끊김 감소)
- vp_threshold 0.03→0.01 (아티팩트 게이트 보수화)

### v49.9 — 최종 정합성 검증
- epochs 기본값 handler 200→150 수정
- f0_autotune 전체 경로 구현 (server.py↔RunPod↔UI)
- protect 주석/도움말 0.33→0.40 현행화

### 핵심 파라미터 (v49.9 최종)
- index_rate: 0.45 (ko:0.55, en:0.35)
- protect: 0.40, filter_radius: 3 (ko:4)
- hop_length: 128, f0_autotune: True (0.3)
- epochs: 150, batch: 8, SR: 40kHz
- language: auto/ko/en
- 1.2kHz -1.0dB (비음 공명), 8kHz -0.8dB (금속음)

### 다음 할 일
1. Docker 재빌드 (karaoke 모델 + Applio 청크 패치)
2. v49 파라미터로 3곡 재변환 테스트
3. My Voice v42 재학습: epochs 150, batch 8
4. (선택) Seed-VC 고음 증강
