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

## Current Status (2026-04-07)
- **v39 모델 학습 완료** + **v43 후처리 최적화 (최신)**
- R2 전체: 9,207개 파일, 28.6GB
- **v43 변환 파라미터**: index 0.30, rms 0.0, protect 0.33, filter_radius 3
- **v43 후처리 체인**:
  - highpass 70Hz → 300Hz -1.5dB → 600Hz -1.0dB → 3kHz +2.0dB (presence)
  - 6.5kHz -1.0dB/w=0.3 (디에서) → loudnorm -14 LUFS (LRA=20)
- **v41→v43 이력**: v41 EQ 축소, v42 발음 EQ 제거, v43 가래/presence 해결
- 고음곡 프리셋 추가 (pitch -3 권장, protect 0.25, filter 5)
- 테스트 48/48 통과
- **CVE-2025-32434**: PyTorch 2.1.0 RCE 취약점 인지 — 2.6.0+ 업그레이드 예정

## Parameters (v41)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: 200, Batch: 8, Sample rate: 40kHz
- F0: RMVPE + FCPE, Embedder: ContentVec (768-dim)
- index_rate: 0.30, rms_mix_rate: 0.0, protect: 0.33, filter_radius: 3
- Overtraining detector: 50 epoch threshold
- vocal_blend: 10% (원본 보컬 블렌딩)
- 학습 데이터: 음원 9개 (장홍권 기존 녹음물)

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
- **v41 5에이전트 종합 음질 개선** (아래 상세)

## v41 음질 개선 (2026-04-07, 5에이전트 종합)
- **분석 방법**: 오디오 분석 + 한국 커뮤니티(아카라이브/디시) + 글로벌(Reddit/AI Hub/GitHub) + 기술 논문 + 코드 리뷰
- **3곡 변환 분석 결과**:
  - Breaking Through: LUFS -10.95, TP +0.45dB (클리핑!)
  - 기다릴게: LUFS -9.57, TP +0.42dB, Peak count 152 (심각)
  - comethru: LUFS -14.72, 36 silence gaps (끊김)
- **핵심 발견**:
  - 2.5kHz/-2.5dB/w=0.8 EQ가 발음 포먼트(F3/F4) 파괴의 주원인
  - 44.1kHz 하드코딩 버그: 48kHz→40k→44.1k (48k로 복원 안됨)
  - 이중 리미터(후처리+믹스)가 펌핑/기계음 유발
  - filter_radius=5의 11프레임 미디언이 고음 F0 지연 유발
- **커뮤니티 합의**: protect 0.33 balanced, de-esser 필수, 화음 별도 분리 변환

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
