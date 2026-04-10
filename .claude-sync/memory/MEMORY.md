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

## Current Status (2026-04-10)
- **v49.9 전면 개편 완료** (19개 커밋, 18개+ 에이전트)
- 9가지 문제 해결: 고음끊김, 치찰음, 발음, 화음, 비음, 끊김, 과적합, 파라미터버그, Dockerfile
- 테스트 48/48 통과
- **CVE-2025-32434**: PyTorch 2.1.0 RCE 취약점 인지
- YingMusic-SVC: CC-BY-NC-4.0 → 학습 데이터 증강용으로만 사용

## Parameters (v49.9 최종)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: **150**, Batch: **8**, Sample rate: 40kHz
- F0: RMVPE + **f0_autotune=True** (strength=**0.3**), Embedder: ContentVec (768-dim)
- index_rate: **0.45** (한국어 0.55 / 영어 0.35), rms_mix_rate: 0.0
- protect: **0.40**, filter_radius: **3** (한국어 4), hop_length: **128**
- language: **auto/ko/en** (한/영 EQ 분리)
- vocal_blend: 0%, post_reverb: 0.0
- split_audio: >180초
- 후처리 EQ: 1.2kHz -1.0dB (비음) + 8kHz -0.8dB (금속음) 공통
- 학습 전처리 디에싱: 8.5kHz -1.0dB (v49.3 보수화)
- 리드/백킹 분리: mel_band_roformer_karaoke (SDR 10.20)
- Applio 청크: x_center=60, x_max=65 (끊김 감소)
- 학습 데이터: 9개 소수정예 (44.9분, 노래 100%, 음역 4.1옥타브)
- 증강 도구: Seed-VC + Rubber Band/WORLD 피치시프트

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
- **v45 더블링 제거 + 치찰음 개선**
- **v49 고음/치찰음/발음 전면 개선** (아래 상세)

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

## v49 고음/치찰음/발음 전면 개선 (2026-04-09)
- **문제**: v41 모델 변환 시 고음/가성 끊김·삑사리, 치찰음/기계음, 발음 부정확
- **근본 원인** (커뮤니티 리서치 + Applio 공식문서):
  - f0_autotune=False → 가성 피치 불안정 (Applio: 노래에 권장)
  - hop_length=64 → 노이즈 추적→삑사리 (커뮤니티: 128 표준)
  - 3kHz +1.0dB → 치찰음 증폭 (HiFi-GAN 이미 충분)
  - 300Hz/600Hz EQ → 한국어 비음 포먼트 파괴
  - 한/영 동일 파라미터 → 자음 특성 차이 무시
- **수정사항**:
  - f0_autotune: False→True (strength 0.6, 비브라토 보존)
  - hop_length: 64→128, filter_radius: 2→3, protect: 0.33→0.40
  - 3kHz presence boost 완전 제거
  - language 파라미터 신규 (ko/en/auto)
  - 한국어: 300Hz/600Hz EQ 제거 (비음 보호), 영어: 경미한 감쇄만
  - split_audio: 300→180초, 프리셋 12개 전면 재설계

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
