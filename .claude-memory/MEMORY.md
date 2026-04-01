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

## Current Status (2026-04-01)
- v35 모델 학습 완료
- KLM49_HFG (한국어 노래) + RIN_E3 (영어 팝/다국어) 이중 pretrained 지원
- PC 간 클라우드 동기화 (Cloudflare R2 백업/복원) 구현 완료
- 테스트 43/43 통과 (pytest)
- ruff + bandit 정적 분석 클린
- 교차 검증 완료: server↔handler↔HTML 기본값/주석/툴팁 100% 동기화
- **CVE-2025-32434**: PyTorch 2.1.0 RCE 취약점 인지 — 2.6.0+ 업그레이드 예정 (Applio 호환성 검증 필요)

## Training Parameters (v35 — 한국어 커뮤니티 최적값)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: 150, Batch: 4, Sample rate: 40kHz
- F0: RMVPE, Embedder: ContentVec (768-dim)
- index_rate: 0.35, rms_mix_rate: 0.25, filter_radius: 3
- Overtraining detector: 50 epoch threshold
- 학습 데이터: 음원 9개 (장홍권 기존 녹음물)

## Workflow Rules (하네스 제약)
- **코드 수정 후 반드시 git commit + git push**
- **코드 변경 후 반드시 `python3 -m pytest tests/ -v` 실행**
- **코드 변경 시 관련 주석/설명/툴팁도 현행화**
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
- v15 vocal pitch pre-shift 비활성화: librosa STFT phase vocoder가 formant 미보존 → 이중 적용 시 기계음
- v35 KLM49_HFG 도입: 한국어 노래 최적화 pretrained (기존 TITAN 교체)

## 코드 감사 이력 (2026-04-01)
- 보안: path traversal 2건, XSS 6건, SQLite 리소스 누수 1건 수정
- 버그: 업로드 race condition, JSON 파싱 오류 타입 수정
- 테스트: 34→43개 (다운로드/취소/청크업로드/전처리리셋 등 추가)
- 데드코드: pitch pre/post-shift 40줄 제거, 미사용 fixture/import 정리
- Dockerfile: CVE-2025-32434 경고 문서화, 모델 파일 크기 검증 추가

## Detailed Memories
- [feedback_no_unnecessary_questions.md](feedback_no_unnecessary_questions.md) — 인프라 질문 금지
- [feedback_harness_engineering.md](feedback_harness_engineering.md) — 하네스 엔지니어링 + Full Agent Team 모드
- [project_v13_pipeline_fixes.md](project_v13_pipeline_fixes.md) — v29 기계음 원인 + v13 수정
- [project_v23_hpss_revert.md](project_v23_hpss_revert.md) — HPSS 실패 + 향후 로드맵
- [project_code_audit_2026_04.md](project_code_audit_2026_04.md) — 전체 코드 감사 결과
