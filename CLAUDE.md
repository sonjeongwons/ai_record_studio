# CLAUDE.md — AI Voice Studio (전체 맥락 문서)

> **이 파일은 Claude Code가 프로젝트를 열 때 자동으로 읽힙니다.**
> 어떤 PC, 어떤 대화창에서든 `git pull`만 하면 모든 맥락이 복원됩니다.
> 상세 메모리는 `.claude-sync/memory/` 디렉토리에 별도 보관되어 있으며,
> 필요 시 해당 파일들을 직접 Read하여 참조하세요.
>
> 🎨 **UI/프론트엔드 코드 작성 시**: `UI_DESIGN_STANDARD.md`를 반드시 먼저 Read하고,
> 자가 비판 루프(Generator → Evaluator → 창의적 도약)를 거친 뒤 코드를 제출할 것.

---

## 1. 프로젝트 컨텍스트

이 프로젝트는 **AI 보이스 클로닝 스튜디오**입니다.
전직 가수(장홍권)의 기존 녹음물(9곡)로 AI 보이스 모델을 학습시켜 새 앨범을 제작하는 것이 목표이며,
이후 상용 제품으로 다른 고객에게도 판매할 예정입니다.

- **`HANDOFF.md`**: 전체 아키텍처 결정, 모델 선택 이유, 비용 분석 상세 문서
- **GitHub**: sonjeongwons/ai_record_studio

## 2. 기술 스택

| 영역 | 기술 |
|------|------|
| 보컬 AI | RVC v2 (Applio) — **SVC(Singing Voice Conversion), TTS 아님!** |
| GPU 클라우드 | RunPod Serverless (RTX 4090) |
| 백엔드 | Python FastAPI + SQLite (WAL 모드) |
| 프론트엔드 | 순수 HTML/CSS/JS (프레임워크 없음) |
| 데스크톱 앱 | PyWebView + PyInstaller (.exe 패키징) |
| 배포 | 단일 .exe + RunPod Docker 이미지 |
| 로깅 | Python `logging` (콘솔 + server.log, RotatingFileHandler 5MB x 3) |
| 테스트 | pytest + FastAPI TestClient — **48개** |
| CI/CD | GitHub Actions (Docker 빌드 + pytest) |
| 정적분석 | ruff (린터) + bandit (보안) |

## 3. 현재 상태 (2026-04-26)

- ✅ 클라이언트 (server.py + index.html) 완성, API 테스트 **50/50 통과**
- ✅ RunPod Serverless Handler 구현 완료
- ✅ PC 간 클라우드 동기화 (Cloudflare R2 백업/복원, 중복 스킵)
- ✅ 보안 감사 완료 (44건+ 수정)
- ✅ **v55 음질 종합 개선 (2026-04-21)**:
  - EQ 감쇠량 65% 축소: 영어 -6.4dB → -2.1dB, 한국어 -5.0dB → -1.5dB (순수 감쇠)
  - 보컬 2-pass loudnorm → 피크 정규화 (-1 dBFS): 다이나믹 레인지 보존
  - LRA=11 → LRA=20 전체 통일 (5곳): 다이나믹 압축 해소
  - 학습 파이프라인 44.1kHz 하드코딩 제거 (6곳): 원본 SR 보존
  - 하이패스 80Hz 전체 통일 (학습 50/60Hz → 80Hz)
  - 학습 PCM_16 → PCM_24 (정밀도 향상)
  - 세그먼트 폴백 10s 하드코딩 → segment_max 동적 사용
  - silent except → log.debug, sf.info() 최적화 (메모리 절약)
- ✅ **v62 바이패스 전면 재설계 (2026-04-26)**:
  - `_detect_gender_bypass_segments`: 200Hz→280Hz (남성 팔세토 범위 제외), frame 0.5s→0.25s, min_dur 1.0s→0.4s, 마지막 블록 truncation 버그 수정
  - `_detect_polyphonic_regions`: HPS 다중 피치 감지 추가, flatness 0.12→0.08, full_vocal_path(리드/백킹 분리 전) 우선 사용
  - `_detect_falsetto_regions` 신규: F0>350Hz + 2초 윈도우 std>1.8반음 조건 불안정 고음 구간 → 원본 유지
  - `_blend_with_bypass`: fade_ms 80→200ms, adjacent segment overlap 버그 수정, rvc_ref/org_ref 불변 참조
  - 파이프라인: full_vocal_path 저장(리드/백킹 분리 전), falsetto_bypass 파라미터 추가(기본 True), harmony_bypass 기본 True로 변경, bypass 원본 소스 full_vocal_path 우선 사용
  - filter_radius 기본값 2→3 (미디언 F0 스무딩 재활성화, 팔세토 안정화)
  - 테스트 50/50 통과
- ✅ **v60 화음/여성보컬 바이패스 + 파라미터 동기화 (2026-04-25)**:
  - `_detect_polyphonic_regions()`: spectral flatness 기반 화음 구간 자동 감지
  - `_detect_gender_bypass_segments()`: pYIN F0 기반 여성 보컬 구간 자동 감지 (200Hz 임계)
  - `_blend_with_bypass()`: 바이패스 구간 원본 오디오 교체 + 80ms 크로스페이드
  - `harmony_bypass`, `female_bypass` API 파라미터 추가 (server.py + index.html + runpod_handler)
  - v57 파라미터 runpod_handler.py 동기화: index=0.50, protect=0.50, filter=2, rms=0.15, strength=0.2
  - 테스트 50/50 통과
- ✅ **v57 5곡 나노분석 기반 파라미터 최적화 (2026-04-23)**:
  - f0_autotune_strength: 0.4→0.2 (피치 상방편향 +2~3반음 교정)
  - protect: 0.40→0.50 (파열음 70회/치찰음 3.4% 개선)
  - highshelf 10kHz: +1.5→+0.8dB (Monster/Breaking 고역 1.4x 과다 해소)
  - server.py + index.html 기본값/프리셋 전면 동기화 완료 (2026-04-24)
  - GHA Docker 캐시 손상 수정 (scope v2→v3, invalid JSON 에러 해결)
- ⚠️ **CVE-2025-32434**: PyTorch 2.1.0 RCE — 2.6.0+ 업그레이드 예정

## 4. 변환 파라미터 (v55 — 최신)

| 파라미터 | 값 | 변경 이력 |
|----------|-----|-----------|
| Pretrained | **KLM49_HFG** (한국어) / **RIN_E3** (다국어) | |
| Epochs | **150** (기본값), Batch: **8**, SR: 40kHz | |
| F0 | RMVPE (+ **f0_autotune=True, strength=0.2**) | v57: 0.4→0.2 (피치 상방편향 교정) |
| **index_rate** | **0.50** (기본값, 서버/HTML 기준) | v55: 0.45→0.50 |
| **rms_mix_rate** | **0.15** | v55: 0.20→0.15 |
| **protect** | **0.50** | v55: 0.40→0.50 |
| **filter_radius** | **3** | v62: 2→3 (미디언 F0 스무딩 재활성화, 팔세토 안정화) |
| **hop_length** | **128** | v49: 64→128 |
| **vocal_blend** | **0%** (비활성) | v45: 더블링 원인 제거 |
| **language** | **auto/ko/en** | v49: 한/영 EQ 분리 |
| 보컬 분리 | **BS-Roformer → Demucs** 폴백 | SDR 12.9 SOTA |
| 후처리 EQ (공통) | agate + HPF **80Hz** + **800Hz -1.0** + **5kHz -0.7** + 디에서(6.5k -1.0, 9k **-0.3**) + **10kHz +0.8 Air** + **피크 정규화** | v57: Air 1.5→0.8 |
| 후처리 EQ (영어) | 공통 + 300Hz -0.3 + 600Hz -0.3 (한국어는 생략) | v55 |
| 보컬 정규화 | **피크 정규화 (-1 dBFS)** (2-pass loudnorm 제거) | v55 |
| LRA | **LRA=20** (5곳 통일, 다이나믹 레인지 보존) | v55: LRA=11→20 |
| **백킹 보컬** | **완전 제거** (리드 보컬만 사용) | v54: 화음 발음 뭉개짐 해소 |
| 믹스 후처리 | 리미터 0.89 + **원곡 LUFS 매칭** (LRA=20, 2-pass) | v55 |
| MR EQ | 800Hz -1.5 + 2kHz -2.0 + 3.5kHz -1.5 | v45: 보컬 블리드 감쇄 |
| 리버브 | 4탭/0.55 decay (비활성 기본) | |
| 샘플레이트 | 원본 SR 보존 (preprocess 44.1kHz 제거, Demucs=44.1k 유지) | v55 |
| split_audio | **>180초** (3분) 분할 | v49: 300→180초 |
| 학습 highpass | **80Hz** (preprocess/gt/16k 모두) | v55: 50/60Hz→80Hz 통일 |
| 학습 PCM | **PCM_24** (peak normalization) | v55: PCM_16→PCM_24 |
| 세그먼트 폴백 | **segment_max 기반** (10s 하드코딩 제거) | v55 |

## 5. AI 모델 사용 규칙 (필수 — 예외 없음)

| 작업 유형 | 사용 모델 |
|----------|---------|
| 분석, 아키텍처 설계, 트레이드오프 검토, 계획 수립, 리서치 | **Opus** |
| 코드 작성, 버그 수정, 파일 편집, 테스트 작성, 리팩토링 | **Sonnet** |

- Agent tool 사용 시 `model` 파라미터 **반드시 명시**: 설계 에이전트 `model: "opus"`, 코딩 에이전트 `model: "sonnet"`
- 설계→구현 순서 작업: Plan 에이전트(opus) 먼저 실행 → Code 에이전트(sonnet) 실행

---

## 6. 컨벤션

- Python 3.10+, 타입힌트 사용
- 주석: 한국어/영어 혼용
- UI: 한국어, 다크 테마 (`#0a0a14` 배경, `#8b5cf6` 바이올렛, `#7a7a90` 텍스트)
- 라이선스/과금 시스템은 나중에 추가 (현재 제외)
- RunPod 클라이언트 변수명: `runpod_client` (runpod 모듈과 충돌 방지)
- DB 경로는 상대 경로 저장 (이식성), `voice_models(name)`에 UNIQUE 인덱스
- **코드 수정 후 반드시 `git commit` + `git push`까지 완료**
- **코드 변경 후 반드시 `python3 -m pytest tests/ -v` 실행하여 회귀 확인**
- **코드 변경 시 관련 주석/설명/툴팁도 현행화 필수**

## 6. 하네스 엔지니어링 원칙

이 프로젝트는 **Martin Fowler의 Harness Engineering** 방식으로 운영됩니다.

- **Human On the Loop** — 사용자는 방향(Why), 에이전트가 실행(How) 자율 수행
- **Full Agent Team 모드** — 모든 에이전트팀 풀 가동, 시간/토큰 제한 없이 철저히 진행
- 에이전트끼리 역할 분담: Plan → Code → Test → Review
- 에이전트 간 교차 검증 필수 (자기 검증 X)
- 코드 변경 완료 시 **자동으로 git commit + push** (매번 사용자 확인 불필요)
- 결과물이 불만족이면 출력물이 아니라 하네스(프로세스/제약)를 개선
- 외부 URL, 검색, 커뮤니티, 공식문서, 도구/라이브러리 자유롭게 사용 가능
- LLM 판단보다 린터, 테스트, 실제 실행 결과를 우선
- 독립적 작업은 반드시 병렬 Agent 호출
- 2-3회 실패 시 사용자에게 에스컬레이션

## 7. 하네스 제약 규칙

- 모든 API 변경은 테스트 추가/수정 필수 (`tests/test_api.py`)
- silent `except Exception: pass` 금지 — 최소 `logger.debug()` 포함
- 새 `print()` 사용 금지 — `logger.info/warning/error/critical` 사용
- 장함수(100줄+) 추가 금지 — 헬퍼 함수로 분리
- **이미 설정된 인프라에 대해 불필요한 확인 질문 하지 말 것**

## 8. 인프라 (이미 설정 완료 — 묻지 말 것)

- GitHub Actions: Docker 자동 빌드 (Dockerfile/runpod_handler.py 변경 시)
- GitHub Actions: pytest 자동 실행 (server.py/tests/ 변경 시)
- Docker registry: `ghcr.io/sonjeongwons/ai_record_studio`
- RunPod endpoint: 설정 완료 (`config.json`)
- Cloudflare R2: 대용량 파일 전송 + PC간 동기화

## 9. 프로젝트 이력 (핵심 결정사항)

### v22 HPSS 실패 → v23 리버트
HPSS harmonic 추출이 보컬 자음/숨소리를 percussive로 분류해 제거 → 품질 치명적 파괴.
v23에서 harmonyFilter=0 비활성, index_rate 0.35→0.50 상향.
> 상세: `.claude-sync/memory/project_v23_hpss_revert.md`

### v13 파이프라인 수정
v29 모델의 기계음 원인 진단: asetrate 피치시프트(치명적), SLICE_DURATION 3.5→5.0s, 고음 오버샘플링 축소, adeclick threshold 조정.
> 상세: `.claude-sync/memory/project_v13_pipeline_fixes.md`

### v15 pitch pre-shift 비활성화
librosa STFT phase vocoder가 formant 미보존 → 이중 적용(pre+post) 시 기계음 유발. 코드 제거 완료.

### v35 KLM49_HFG 도입
한국어 노래 최적화 pretrained (기존 TITAN 교체). RIN_E3는 영어 팝/다국어용으로 병행.

### 2026-04-01 코드 감사
4개 에이전트 병렬 감사: 보안 9건(path traversal, XSS, race condition), 버그 3건 수정. 테스트 34→43. CVE-2025-32434 인지.
> 상세: `.claude-sync/memory/project_code_audit_2026_04.md`

### 2026-04-02~03 v36~v40 음질 개선
rms=0, BS-Roformer, loudnorm, 보컬 블렌딩, agate/adeclick 제거 등.

### 2026-04-07 v41 종합 음질 개선 (5에이전트)
44.1kHz 하드코딩 버그 수정, EQ 감쇠 65% 축소 (발음 2-4kHz 보존), 이중 리미터 해소,
filter_radius 5→3, protect 0.35→0.33, 믹스 리미터 0.89 (-1dBTP).
> 상세: `.claude-sync/SESSION_CONTEXT.md`

### v42 EQ 극한 최소화 + v43 발음/가래 해결
v42: 2.8kHz EQ 제거 (발음 F3/F4 -4.4dB 파괴 주범), 디에서 축소.
v43: 300-600Hz -2.5dB (가래 해결) + 3kHz +2.0dB (presence 복원) + 고음곡 프리셋 (pitch -3).

### v45 더블링 제거 + 치찰음 개선 (2026-04-07)
**핵심 문제**: 변환 목소리가 여러명이 부르는 것처럼 중첩됨 + 치찰음/기계음.
**원인 진단**:
  - vocal_blend 10-20%가 원본+변환 보컬 중첩 → 더블링 효과
  - aecho 8탭/0.88 decay가 코러스 효과 생성
  - Demucs 보컬 블리드가 MR에 잔류 → 이중 보컬
  - de-esser 부족 + presence +2.0dB가 치찰음 강조
**수정**: vocal_blend→0, 리버브 4탭/0.55, de-esser 2단(5kHz+8kHz),
  presence +1.0dB, MR 블리드 감쇄 EQ 3대역, 리미터 level=disabled,
  프리셋 전면 재설계 (protect 0.50, filter 2).

### v49 고음/치찰음/발음 전면 개선 (2026-04-09)
**핵심 문제**: v41 모델 변환 시 고음/가성 끊김, 치찰음/기계음, 발음 부정확.
**근본 원인 진단** (커뮤니티 리서치 + Applio 공식문서 + 코드 분석):
  - f0_autotune=False → 가성 피치 불안정 시 보정 없음 (Applio: 노래에 권장)
  - hop_length=64 → 과도한 피치 추적이 노이즈 추적→삑사리 (커뮤니티: 128 표준)
  - 3kHz +1.0dB presence → 치찰음 증폭 주범 (HiFi-GAN 이미 충분)
  - 300Hz/600Hz EQ → 한국어 비음(ㄴ/ㅁ/ㅇ) 포먼트 파괴
  - 한/영 동일 파라미터 → 자음 특성 근본 차이 무시
**수정**:
  - f0_autotune=True, strength=0.6 (비브라토 보존)
  - hop_length 64→128, filter_radius 2→3, protect 0.33→0.40
  - 3kHz presence boost 제거, 한/영 EQ 분리 (language 파라미터 신규)
  - 한국어: 300Hz/600Hz EQ 제거 + filter 4, 영어: 경미한 감쇄만
  - split_audio 임계값 300→180초 (경계 아티팩트 감소)
  - 전 프리셋 v49 전면 재설계 (12개 프리셋)

### v54 Spin V2/Korean HuBERT embedder + BS PolarFormer + Seed-VC (2026-04-12)
**구현**: 커뮤니티 리서치 로드맵 전체 구현
  - Dockerfile: Spin V2 + Korean HuBERT + BS PolarFormer 모델 사전 캐시
  - Embedder 선택 시스템: contentvec/spin/korean-hubert-base 3종 (학습+추론)
  - UI: 학습 탭 임베더 드롭다운, 변환 API embedder_model 파라미터
  - Seed-VC v54: 노래 전용 모델 + 파인튜닝 + time-stretch 변형
  - 주의: 학습/추론 동일 embedder 필수, 재학습 필요

### v54 백킹 제거 + 원곡 LUFS + 부밍 억제 + RMVPE 패치 (2026-04-12)
**분석**: v53 변환 4곡(Breaking/comethru/플레이브/Monster) 정밀 스펙트럼 분석
**핵심 문제**:
  - 백킹 보컬이 화음 발음 뭉개짐/기계음 중첩의 주범 → 완전 제거
  - -14 LUFS 하드코딩 → 원곡(-6~-12) 대비 2-7.6 LUFS 저하 → 원곡 매칭
  - v53 EQ가 8-16kHz -7~-12dB 과도 손실 → 감쇄량 축소 + Air 복원
  - 500-2kHz +3.1dB 보코더 부밍 → 800Hz -1.5dB 억제
  - RMVPE 32프레임 제한 → 가성 피치 불안정 근본 원인
**수정**:
  - 백킹 보컬 완전 제거 (리드+MR only)
  - 원곡 LUFS 측정→매칭 정규화
  - EQ: 800Hz -1.5dB, 5kHz -1.0, 8kHz -0.5, 10kHz +1.5 Air
  - Dockerfile: x_center 60→90, RMVPE 32→94프레임

### v53 분석 기반 음질 전면 개선 (2026-04-12)
**분석**: v52 변환 3곡 정밀 스펙트럼/LUFS 분석 + 커뮤니티 리서치 (Reddit/HuggingFace/Arca.live)
**핵심 문제** (분석 데이터):
  - 치찰음 스파이크 75-100% 증가 (4-8kHz 유일 증폭 대역)
  - 변환곡 2-4 LUFS 낮음, LRA 들쑥날쑥
  - 가성 비브라토 과도 평탄화 (autotune 0.6이 원인)
  - 1-4kHz 발음 대역 3-5dB 손실
**수정**:
  - HPF 70→80Hz, 5kHz -1.5dB (보코더 금속성), 2단 디에서 (6.5k+9k)
  - 믹스 최종단 LUFS -14 정규화 (2-pass loudnorm)
  - f0_autotune_strength 0.6→0.4, filter_radius 2→3, rms_mix_rate 0.1→0.20
  - 프리셋 11개 전면 재설계 (화음 filter 5, 한국어 filter 4)

### v49.5 리드/백킹 분리 + 화음 처리 (2026-04-09)
**핵심 문제**: RVC가 화음(리드+백킹) 전부를 동일 음색으로 변환 → 부자연스러운 화음.
**해결**: 3단계 파이프라인:
  1. `_separate_lead_backing()`: 보컬 스템에서 리드/백킹 추가 분리 (mel_band_roformer_karaoke)
  2. 리드 보컬만 RVC 변환, 백킹/화음은 원본 유지
  3. RVC 리드 + 원본 백킹(0.7 볼륨) + MR → 3트랙 믹싱

### v49.1 에폭 최적화 + 학습 디에싱 (2026-04-09)
- epochs: 200→150 (45분 데이터에 250 과적합→치찰음)
- batch_size: 4→6 (AI Hub: >30분=8, <30분=4, 44.9분→중간값)
- 학습 전처리 디에싱 추가: 6kHz -2dB + 8.5kHz -1.5dB (모델이 치찰음을 목소리 특성으로 학습 방지)
- Seed-VC 증강 도구 추가 (tools/seed_vc_augment.py): 고음역 학습 데이터 생성
- WORLD 보코더 피치 증강 도구 추가 (tools/pitch_augment.py): 포먼트 보존 피치시프트

### v55 음질 종합 개선 재점검 (2026-04-21)
**핵심 문제**: Phase 1.5~5 구현 감사 결과, 대부분 미구현/과도 적용 확인.
**수정사항**:
  - EQ: 언어별 5kHz 중복 컷 제거, 800Hz -1.5→-1.0, 1.2kHz 컷 제거, 5kHz -1.0→-0.7, 8kHz 컷 제거, 9kHz -0.5→-0.3
  - 보컬 2-pass loudnorm → volumedetect 기반 피크 정규화 (-1 dBFS, 카스케이드 제거)
  - LRA=11 → LRA=20 전체 5곳 (LUFS 매칭 다이나믹 압축 해소)
  - 전처리/학습 44.1kHz 하드코딩 6곳 제거 (소스 SR 보존)
  - 학습 highpass 50/60Hz → 80Hz 전체 통일 (gt/16k/MP3 경로 포함)
  - 피크 정규화 PCM_16 → PCM_24 (학습 데이터 정밀도 향상)
  - 세그먼트 폴백 10s 하드코딩 → segment_max 동적 값 사용
  - silent except → log.debug, sf.info() 최적화
  - SEGMENT_MIN/MAX 상수 (데드 코드) 제거

### v57 5곡 나노분석 기반 파라미터 최적화 (2026-04-23)
**분석**: My Voice v44 모델 + 변환곡 5개 (mixed_*57.wav) 나노 단위 분석
**핵심 발견**:
  - 전 곡 피치 상방편향 +1.4~+3.0반음 (f0_autotune strength=0.4가 원인)
  - 모델 과적합 신호: attn_layers.3.conv_k.bias absmax=6.61 (정상의 10배)
  - flow.res_skip_layers.2.bias 완전 사망 (std=0.001) → 피치 흐름 레이어 고장
  - FAISS participation ratio=45/768 (실효 차원 협소 → AI스러움)
  - nprobe=1 (검색 품질 낮음 — Applio 내부 파라미터로 직접 패치 불가)
  - Monster/Breaking Through: 5kHz+ 에너지 1.4x 과다 (highshelf 원인)
**수정**:
  - f0_autotune_strength: 0.4→0.2 (피치 상방편향 교정)
  - protect: 0.40→0.50 (파열음 70회/치찰음 3.4% 개선)
  - highshelf 10kHz: +1.5→+0.8dB (고역 과다 해소)
  - server.py 기본값 5개 CLAUDE.md v55와 동기화 (index_rate, rms_mix_rate, filter_radius)
**재학습 권고**: epoch 100→70~75 (loss_g 4.0 이상 구간에서 best_epoch), Spin V2 embedder

### 향후 로드맵
- PyTorch 2.6.0+ 업그레이드 (CVE 대응)
- ✅ Seed-VC v54 업데이트 완료 (노래 전용 모델 + 파인튜닝 + time-stretch)
- ✅ **Spin V2 embedder**: Docker 이미지에 추가, UI 선택 가능 (재학습 필요)
- ✅ **Korean HuBERT embedder**: Docker 이미지에 추가, UI 선택 가능 (재학습 필요)
- ✅ **BS PolarFormer**: Docker 이미지에 추가 (BS-Roformer 후속)
- ✅ **RMVPE 프레임 버퍼**: 32→94 프레임 패치 (가성 안정화)
- **다음 단계**: Spin V2 vs Korean HuBERT vs ContentVec A/B 테스트 (3개 모델 학습 비교)
- Codename-RVC-Fork-3 검토: PESQ/SI-SDR 검증 메트릭, Ranger21 옵티마이저

## 10. 주요 명령어

```bash
python server.py                    # 개발 서버
python3 -m pytest tests/ -v         # 테스트 (48개)
python app.py                       # 데스크톱 앱
python build_exe.py                 # EXE 빌드
docker build -t ai-voice-studio .   # Docker 빌드
```

## 11. 파일 구조

```
ai_record_studio/
├── CLAUDE.md              ← 이 파일 (자동 로드, 전체 맥락)
├── HANDOFF.md             ← 아키텍처/결정사항 상세 문서
├── .claude-sync/memory/        ← 상세 메모리 (git 포함)
│   ├── MEMORY.md
│   ├── feedback_harness_engineering.md
│   ├── feedback_no_unnecessary_questions.md
│   ├── project_v13_pipeline_fixes.md
│   ├── project_v23_hpss_revert.md
│   └── project_code_audit_2026_04.md
├── .claude/settings.json  ← Claude Code 프로젝트 설정 (git 포함)
├── server.py              ← FastAPI 백엔드 (~3300줄, 35+ API)
├── static/index.html      ← 웹 UI (~8400줄, 다크 테마, 4탭)
├── runpod_handler.py      ← RunPod GPU 핸들러 (~3500줄)
├── Dockerfile             ← RunPod Docker 이미지
├── app.py                 ← 데스크톱 앱 (PyWebView)
├── build_exe.py           ← PyInstaller EXE 빌드
├── tests/                 ← API 테스트 (48개)
│   ├── conftest.py
│   └── test_api.py
└── .github/workflows/     ← CI/CD
```

## 12. API 엔드포인트 (전체)

```
GET  /                              → 웹 UI
GET  /api/config                    → RunPod/R2 설정 조회
POST /api/config                    → RunPod 설정 저장
POST /api/config/r2                 → R2 스토리지 설정 저장
POST /api/config/download-folder    → 다운로드 폴더 설정
POST /api/save-to-folder            → 파일을 다운로드 폴더에 복사
POST /api/upload                    → 파일 업로드 (다중)
POST /api/upload/chunk              → 청크 업로드 (대용량, 최대 2GB)
GET  /api/files                     → 업로드 파일 목록
DELETE /api/files/{id}              → 파일 삭제 (soft delete)
POST /api/preprocess                → 전처리 시작 (RunPod)
GET  /api/preprocess/status         → 전처리 상태 조회
GET  /api/preprocess/download/{fn}  → 전처리 파일 다운로드 (MR/보컬)
DELETE /api/preprocess              → 전처리 결과 전체 삭제
POST /api/preprocess/reset          → 선택 파일 전처리 초기화
POST /api/train                     → 학습 시작 (RunPod)
POST /api/convert                   → 변환 시작 (RunPod)
GET  /api/jobs/{id}                 → 작업 상태 조회
GET  /api/jobs/active               → 현재 실행 중 작업
GET  /api/jobs                      → 전체 작업 목록
POST /api/jobs/cleanup              → 작업 정리
POST /api/jobs/{id}/cancel          → 작업 취소
POST /api/jobs/{id}/pause           → [제거됨] 410 Gone
POST /api/jobs/{id}/resume          → [제거됨] 410 Gone
GET  /api/models                    → 모델 목록
DELETE /api/models/{id}             → 모델 삭제 (cascade)
POST /api/models/{id}/rename        → 모델 이름 변경
POST /api/models/{id}/quality       → 품질 점수 업데이트
GET  /api/conversions               → 변환 이력
DELETE /api/conversions/{id}        → 변환 이력 삭제
GET  /api/download/{name}           → 파일 다운로드
GET  /api/stats                     → 대시보드 통계
GET  /api/health                    → 서버/RunPod/DB 상태
GET  /api/system-info               → 시스템 정보
POST /api/sync/backup               → 클라우드 백업 (R2)
POST /api/sync/restore              → 클라우드 복원 (R2)
GET  /api/sync/status               → 동기화 상태 조회
```
