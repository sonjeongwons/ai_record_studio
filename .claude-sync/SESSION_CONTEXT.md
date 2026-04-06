# Session Context — 2026-04-06

이 파일은 다른 PC에서 Claude Code 대화를 이어가기 위한 컨텍스트 요약입니다.
새 세션 시작 시 "SESSION_CONTEXT.md 읽어줘" 라고 하면 됩니다.

## 마지막 작업 상태

### 완료된 작업: v39 음질 최적화 (전 세션에서 완료)
- v39 모델 학습 파라미터 최적화 (글로벌+한국 커뮤니티 종합)
- EQ 타겟 이동, index 0.30, filter 5

### 이번 세션에서 진행한 작업: 자연스러움 극대화 개선

**Phase 1.1** — `runpod_handler.py` `_rvc_preprocess()`: loudnorm → 피크 정규화
- loudnorm=I=-20:TP=-1:LRA=7 제거 → numpy peak normalize -1 dBFS
- 조용한 세그먼트 자동 스킵 (RMS < -40 dBFS)

**Phase 1.2+2.1~2.3** — `_post_process_vocal()` 후처리 체인 전면 개선
- 디에서 추가 (6kHz -3dB), EQ 완화 (-1.5dB→-1.0dB), 새추레이션 완화
- 8-tap 리버브 (소수 간격 7|13|23|31|41|53|67|83ms)

**Phase 2.4~2.5** — `_mix_audio()` 믹스 개선
- MR 주파수 매칭 EQ (800Hz -2dB, 2.5kHz -1.5dB)
- 리미터: attack 10→25ms, release 100→300ms, limit 0.99→0.95

**Phase 2.6** — `_noise_reduce()` 적응형 NR
- SNR 기반: 클린(>-20dB)→스킵, 보통→prop 0.25, 노이즈→prop 0.35

**Phase 3.1** — `_segment_audio()` 세그먼트 경계 10ms 크로스페이드

**Phase 3.3** — 오버트레이닝 감지 임계값 75→50

**Phase 1.3** — RVC 기본값 동기화 (runpod_handler.py, server.py, index.html)
- ⚠️ 참고: 이후 v39 작업으로 기본값이 다시 변경되었음
- 현재 기본값: index 0.30, filter 5, rms 0.0, protect 0.35

**Phase 1.4** — 프리셋 업데이트 + studio 프리셋 추가
- default/falsetto/natural 프리셋 값 업데이트
- studio 프리셋 신규 추가 (깨끗한 원본 최적화)

### 현재 코드 상태
- 테스트: 47/48 통과 (1개 실패: test_reset_preprocess_empty — 422 vs 400 기대값 불일치, 기존 이슈)
- git: main 브랜치, 최신 커밋 `3d3fcee`

## 다음 할 일 (사용자가 요청한 경우)
1. v36 모델로 타겟 3곡 변환 테스트
2. 변환 결과 청취 후 파라미터 미세 조정
3. 테스트 실패 1건 수정 (test_reset_preprocess_empty)

## 타겟 곡 3개
- "01_Breaking Through (4824 Wave).wav" — 영어, 팝/록
- "플레이브 - 기다릴게.mp3" — 한국어, K-POP
- "Jeremy Zucker - comethru ft. Bea Miller.mp3" — 영어, 인디팝

## 핵심 규칙 (CLAUDE.md 참조)
- 코드 수정 후 반드시 git commit + git push
- 코드 변경 후 반드시 pytest 실행
- 인프라(RunPod, GitHub Actions, R2)에 대해 불필요한 확인 질문 금지
- Korean UI, dark theme (#0a0a14 bg, #8b5cf6 violet)
