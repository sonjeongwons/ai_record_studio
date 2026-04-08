# AI Voice Studio - Project Memory

## Project Overview
- AI Voice Cloning Studio for singer 장홍권 (former singer)
- 기존 녹음물로 AI 보이스 모델을 학습시켜 새 앨범 제작 → 이후 상용 제품화 예정
- Tech: RVC v2 (Applio) + RunPod Serverless (RTX 4090) + FastAPI + SQLite + vanilla HTML/CSS/JS
- Desktop: PyWebView + PyInstaller (.exe 패키징, Python 설치 불필요)
- Location: `C:\Users\SDS\ai_record_studio`
- GitHub: sonjeongwons/ai_record_studio

## Key Files
- `server.py` — FastAPI 백엔드 (~3300줄, 35+ API, SQLite WAL, 쓰레드 기반 RunPod 폴링)
- `static/index.html` — 웹 UI (~8400줄, 다크 테마 #0a0a14, 바이올렛 #8b5cf6, 4탭)
- `runpod_handler.py` — RunPod GPU 핸들러 (~3500줄, Demucs 6-stem 분리, RVC 변환, 학습)
- `Dockerfile` — RunPod Docker 이미지 (CUDA 12.1, Applio, Demucs, RMVPE, KLM49+RIN_E3)
- `app.py` — 데스크톱 앱 (PyWebView 래퍼)
- `build_exe.py` — PyInstaller EXE 빌드 스크립트
- `CLAUDE.md` — 하네스 엔지니어링 제약 문서 (린터/테스트 규칙, 컨벤션)
- `HANDOFF.md` — 전체 아키텍처 결정사항, 비용 분석, 기술 선택 이유

## Current Status (2026-04-08)
- **v41 학습 데이터 소수정예 최적화 완료**
- 115개 전수조사 → 9개 소수정예 선정 (44.9분, 노래 100%, 대화 0%)
- 깨끗 72% + MR 28%, 음역 53Hz~555Hz (4.1옥타브)
- v40 모델 해부: F0 상관계수 0.35 근본 원인 = 대화 48% + 고음 3.9%
- 사이버틱/중첩 해결: 전 프리셋 리버브 OFF
- v46 전처리: 무음 세그먼트 자동 제거 + -23 LUFS 정규화
- 테스트 48/48 통과
- **CVE-2025-32434**: PyTorch 2.1.0 RCE 취약점 인지
- YingMusic-SVC: CC-BY-NC-4.0 비상업 라이선스 → 프로덕션 제외

## Parameters (v45)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: 200, Batch: 8, Sample rate: 40kHz
- F0: RMVPE + FCPE, Embedder: ContentVec (768-dim)
- index_rate: 0.30, rms_mix_rate: 0.0, protect: 0.50, filter_radius: 2
- Overtraining detector: 50 epoch threshold
- vocal_blend: 0% (비활성 — 더블링 원인이었음)
- 학습 데이터: mp3record/ 71개 파일 (장홍권 기존 녹음물 + 대화 녹음)

## Workflow Rules (하네스 제약)
- **코드 수정 후 반드시 git commit + git push**
- **코드 변경 후 반드시 `python3 -m pytest tests/ -v` 실행**
- **코드 변경 시 관련 주석/설명/툴팁도 현행화**
- **변경 시 CLAUDE.md + .claude-memory + .claude-sync 3종 동시 현행화**
- silent `except: pass` 금지 → 최소 `logger.debug()` 포함
- 새 print() 금지 → logging 사용
- 장함수(100줄+) 금지 → 헬퍼 분리
- 이미 설정된 인프라에 대해 불필요한 확인 질문 금지

## Important Conventions
- RunPod client variable: `runpod_client` (not `runpod` — 모듈과 충돌 방지)
- Korean UI, mixed Korean/English comments
- Dark theme: #0a0a14 bg, #8b5cf6 violet accent, #7a7a90 text3 (WCAG AA)
- No frameworks — pure HTML/CSS/JS frontend
- Pause/resume 기능 제거됨 (HTTP 410 Gone 반환)
- DB 경로는 상대 경로 저장 (이식성)
- voice_models(name)에 UNIQUE 인덱스

## Infrastructure (이미 설정 완료 — 묻지 말 것)
- GitHub Actions: Docker 자동 빌드 (Dockerfile/runpod_handler.py 변경 시)
- GitHub Actions: pytest 자동 실행 (server.py/tests/ 변경 시)
- Docker registry: ghcr.io/sonjeongwons/ai_record_studio
- RunPod endpoint: 설정 완료
- Cloudflare R2: 대용량 파일 전송 + PC간 동기화

## 프로젝트 이력 (중요 결정사항)
- v22 HPSS → v23 리버트: HPSS harmonic 추출이 자음/숨소리 제거 → 품질 파괴
- v13 파이프라인 수정: asetrate 피치시프트 제거 (기계음 원인), SLICE_DURATION 3.5→5.0s
- v15 vocal pitch pre-shift 비활성화: librosa STFT phase vocoder가 formant 미보존
- v31 48kHz 시도 → 보코더 기계음 악화 → v32에서 40kHz 복원
- v35 KLM49_HFG 도입 + Demucs 보컬분리 전처리 추가 (근본 원인 해결)
- v40 agate/adeclick 제거 (최소 후처리 원칙)
- **v41 5에이전트 종합 음질 개선**
- **v45 더블링 제거 + 치찰음 개선** (아래 상세)

## v45 더블링 제거 (2026-04-07)
- **문제**: 변환 목소리가 여러명이 부르는 것처럼 중첩 + 치찰음/기계음 전체적
- **더블링 원인 진단**:
  - vocal_blend 10-20%가 원본+변환 보컬 중첩 → **주요 원인**
  - aecho 8탭/0.88 decay가 코러스 효과 생성
  - Demucs 보컬 블리드가 MR에 잔류 → 이중 보컬
- **치찰음 원인**: presence +2.0dB가 고역 치찰음 강조, de-esser 1개로 부족
- **수정사항**:
  - vocal_blend: 10-20% → 0% (전 프리셋)
  - 리버브: 8탭/0.88 → 4탭/0.55
  - de-esser: 1단(6.5kHz) → 2단(5kHz 광역 + 8kHz 협역)
  - presence: +2.0 → +1.0dB
  - MR EQ: 2kHz -2.0dB, 3.5kHz -1.5dB 추가 (블리드 감쇄)
  - 리미터: level=enabled→disabled (자동 게인 올림 → 클리핑)
  - 프리셋: protect 0.33→0.50, filter 3→2

## 타겟 곡 3개
- "01_Breaking Through (4824 Wave).wav" — 영어, 팝/록, 48kHz/24bit WAV
- "플레이브 - 기다릴게.mp3" — 한국어, K-POP, 44.1kHz MP3
- "Jeremy Zucker - comethru ft. Bea Miller.mp3" — 영어, 인디팝, 44.1kHz MP3

## 학습 소스 음원 9개 (mp3record/ 폴더)
- 07a51a19, 5a8a4c9a, a2eebb1e, 90d37ee7, 79dde5c2 — MR 포함 (Demucs 분리 필요)
- 장이정-살다가한번쯤 — MR 포함 (Demucs 분리 필요)
- **장이정-살다가테스트** — 깨끗한 보컬 (320kbps, 최고 품질)
- **히스토리_tomorrow** — 깨끗한 보컬 (66초)
- 장이정 학습용 데이터 (1) — 20분, 혼합 (보컬 12분 + MR 8분, Demucs 분리 필요)

## Detailed Memories
- [feedback_no_unnecessary_questions.md](feedback_no_unnecessary_questions.md) — 인프라 질문 금지
- [feedback_harness_engineering.md](feedback_harness_engineering.md) — 하네스 엔지니어링 + Full Agent Team 모드
- [project_v13_pipeline_fixes.md](project_v13_pipeline_fixes.md) — v29 기계음 원인 + v13 수정
- [project_v23_hpss_revert.md](project_v23_hpss_revert.md) — HPSS 실패 + 향후 로드맵
- [project_code_audit_2026_04.md](project_code_audit_2026_04.md) — 전체 코드 감사 결과
- [feedback_memory_maintenance.md](feedback_memory_maintenance.md) — 변경 시 CLAUDE.md + 메모리 3종 동시 현행화 필수
