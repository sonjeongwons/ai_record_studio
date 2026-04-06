# Session Context — 2026-04-07 (v41 세션)

이 파일은 다른 PC/세션에서 Claude Code 대화를 이어가기 위한 컨텍스트 요약입니다.

## 마지막 작업 상태

### 이번 세션에서 진행한 작업

**1. Claude Code PC간 동기화 시스템 구축**
- sync-push.bat / sync-pull.bat 생성
- .claude-sync/ 디렉토리에 메모리+세션 컨텍스트 동기화
- settings.json은 인증 토큰 포함으로 .gitignore 처리

**2. v41 음질 종합 개선 (5에이전트 풀가동)**
- 에이전트팀: 오디오 분석, 한국 커뮤니티(아카라이브/디시), 글로벌(Reddit/AI Hub/GitHub), 기술 논문, 코드 리뷰
- 3곡 변환 결과 FFmpeg 정량 분석 수행

**3곡 변환 분석 결과:**

| 곡 | LUFS | True Peak | LRA | Peak count | 
|----|------|-----------|-----|------------|
| Breaking Through | -10.95 (과대) | +0.45dB (클리핑!) | 4.2 | 59 |
| 기다릴게 | -9.57 (과대) | +0.42dB (클리핑!) | 3.4 | 152 (심각) |
| comethru | -14.72 (정상) | +0.02dB | 5.8 | 3.5 |

**v41 변경사항 (commit dcbef59):**

Phase A — 치명적 버그:
- 44.1kHz 하드코딩 → `_process_sr` (48kHz 보컬/MR SR 불일치 해결)
- post_reverb 백엔드 기본값 0.05→0.0

Phase B — 후처리 체인 (발음 복원 핵심):
- EQ 감쇠 -7.3dB→-2.0dB (65% 감소): 2.5kHz/-2.5dB → 2.8kHz/-1.0dB/w=0.4
- 550Hz, 3.5kHz, 7.5kHz, highshelf +0.5dB 제거
- 6.5kHz 디에서 추가 (치찰음 정적 감쇠)
- 후처리 리미터 제거 (이중 리미터 펌핑 해소)
- loudnorm LRA 11→20 (다이나믹 보존)
- highpass 50→70Hz (파열음 제어)

Phase C — 믹스:
- auto_gain 0.5-6.0→0.7-3.0
- 믹스 리미터 0.95→0.89 (-1dBTP, 클리핑 해결)

Phase D — RVC 파라미터:
- filter_radius 5→3 (고음 F0 지연 해소)
- protect 0.35→0.33 (글로벌 합의)

Phase F — 피치 게이트:
- gain 0.05→0.15 (-26dB→-16dB)

### 현재 코드 상태
- 테스트: 47/48 통과 (1실패: test_reset_preprocess_empty, 기존 이슈)
- git: main 브랜치, 최신 커밋 dcbef59
- Docker 이미지 자동 빌드 중 (GitHub Actions)

### 전처리/학습 재필요 여부
- **전처리**: 불필요 (기존 전처리 데이터 유효)
- **모델 학습**: 불필요 (v39 모델 그대로 사용)
- **변환만 재실행**: v41 변경은 변환 시점에 적용됨

## 다음 할 일
1. Docker 빌드 완료 대기 후 v41로 타겟 3곡 재변환
2. 변환 결과 FFmpeg 분석 재실행 (v40 vs v41 비교)
3. 결과 청취 후 미세 조정

## 타겟 곡 3개
- "01_Breaking Through (4824 Wave).wav" — 영어, 팝/록, 48kHz
- "플레이브 - 기다릴게.mp3" — 한국어, K-POP, 44.1kHz
- "Jeremy Zucker - comethru ft. Bea Miller.mp3" — 영어, 인디팝, 44.1kHz

## v41 후처리 체인 (현재)
```
highpass 70Hz → 300Hz -1.0dB → 2.8kHz -1.0dB/w=0.4 →
6.5kHz -2.0dB (디에서) → (고음모드) → (리버브) →
2-pass loudnorm -14 LUFS (LRA=20)
```

## v41 믹스 체인
```
보컬: auto_gain(0.7-3.0) → stereo
MR: lowshelf 60Hz/-0.8 → 800Hz -1.0 → 2500Hz -1.0
amix → alimiter 0.89 (-1dBTP)
```

## 핵심 파라미터 (v41)
- index_rate: 0.30, rms_mix_rate: 0.0, protect: 0.33, filter_radius: 3
- epochs: 200, batch: 8, SR: 40kHz, F0: RMVPE
- vocal_blend: 10% (프리셋 기본)

## 5에이전트 리서치 핵심 결론
- **한국 커뮤니티**: protect 0.33 적절, de-esser 필수, 화음 별도 분리 변환, RMVPE 1순위
- **글로벌 커뮤니티**: index_rate 0.30 유지(MP3 소스), 오버트레이닝이 #1 원인, 입력 볼륨 증폭 팁
- **기술 논문**: RMVPE+FCPE 최적 확인, 한국어 protect 0.28-0.33, 고주파 선택적 블렌딩(SYKI-SVC)
- **코드 리뷰**: 2.5kHz EQ가 발음 파괴 주원인, 이중 리미터 펌핑, 44.1k 하드코딩 버그

## 핵심 규칙 (CLAUDE.md 참조)
- 코드 수정 후 반드시 git commit + git push
- 코드 변경 후 반드시 pytest 실행
- 인프라에 대해 불필요한 확인 질문 금지
- 변경 시 CLAUDE.md + .claude-memory + .claude-sync 3종 현행화
