# Session Context — 2026-04-06 (PC2 세션)

이 파일은 다른 PC/세션에서 Claude Code 대화를 이어가기 위한 컨텍스트 요약입니다.

## 마지막 작업 상태

### 이번 세션에서 진행한 작업

**1. 시스템 총점검 4차 (v39 기본값 동기화)**
- _rvc_infer() 함수 시그니처 4개 스탈 기본값 수정 (index 0.35→0.30, protect 0.50→0.35, filter 2→5, rms 0.10→0.0)
- HTML index-val 표시 0.40→0.30, protect getVal 폴백 40→35
- epochs select 150→200, batch select 4→8
- 프리셋 desc 4건 + 버튼 title 3건 현행화

**2. 클라이언트 전수 현행화 (35건)**
- 툴팁: protect 기본값 0.40→0.35, filter "기본3"→"기본5", epochs v35→v39
- 학습 가이드: 에폭 150→200, 비용 테이블 batch 8 기준
- 시나리오: 25분/150에폭 → 43분/200에폭
- JS 주석: v35/v36→v39 전면 교체

**3. v40 후처리 전면 개편 (핵심 변경)**
- **agate 제거**: release=80ms 너무 짧음 + range -32dB 너무 강함 → spectral flux 2.11x 펌핑
- **adeclick 제거**: burst=4가 P/T/K 자음 transient 삼킴 → 원본보다 -242개
- **300-600Hz 중역 EQ 추가**: 300Hz -1.5dB + 550Hz -1.0dB (HiFi-GAN 블로트 +68~113%)
- **highshelf 1.0→0.5dB**: 치찰음 +59% 과도 완화
- **MR lowshelf 완화**: 80Hz/-1.5→60Hz/-0.8 (서브베이스 -13~18% 보존)
- 커뮤니티 합의: "최소 후처리가 최선" (AI Hub)

### 현재 코드 상태
- 테스트: 48/48 통과
- ruff: All checks passed
- git: main 브랜치, 최신 커밋

### 전처리/학습 재필요 여부
- **전처리**: 불필요 (기존 전처리 데이터 유효)
- **모델 학습**: 불필요 (v39 모델 그대로 사용)
- **변환만 재실행**: v40 후처리 개선은 변환 시점에 적용됨

## 다음 할 일
1. v39 모델 + v40 후처리로 타겟 3곡 재변환
2. 결과 청취 후 미세 조정
3. Docker 리빌드 (GitHub Actions 자동)

## 타겟 곡 3개
- "01_Breaking Through (4824 Wave).wav" — 영어, 팝/록, 48kHz
- "플레이브 - 기다릴게.mp3" — 한국어, K-POP, 44.1kHz
- "Jeremy Zucker - comethru ft. Bea Miller.mp3" — 영어, 인디팝, 44.1kHz

## v40 후처리 체인 (현재)
```
highpass 50Hz → 300Hz -1.5dB → 550Hz -1.0dB → 2.5kHz -2.5dB →
3.5kHz -1.5dB → 7.5kHz -0.8dB → highshelf 10kHz +0.5dB →
(고음모드) → limiter 0.98 → (리버브) → 2-pass loudnorm -14 LUFS
```

## 핵심 파라미터 (v39/v40)
- index_rate: 0.30, rms_mix_rate: 0.0, protect: 0.35, filter_radius: 5
- epochs: 200, batch: 8, SR: 40kHz, F0: RMVPE
- vocal_blend: 10% (프리셋 기본)

## 핵심 규칙 (CLAUDE.md 참조)
- 코드 수정 후 반드시 git commit + git push
- 코드 변경 후 반드시 pytest 실행
- 인프라에 대해 불필요한 확인 질문 금지
- 변경 시 CLAUDE.md + .claude-memory + .claude-sync 3종 현행화
