# AI Voice Studio - Project Memory

## Project Overview
- AI Voice Cloning Studio for singer 장홍권
- Tech: RVC v2 (Applio) + RunPod Serverless + FastAPI + vanilla HTML/CSS/JS
- Location: `C:\Users\SDS\ai_record_studio`

## Key Files
- `server.py` - FastAPI backend (35+ API endpoints, RunPod client with retry)
- `static/index.html` - Web UI (dark theme, 4 tabs, toast notifications, audio player)
- `runpod_handler.py` - RunPod Serverless GPU handler (preprocess/train/convert)
- `Dockerfile` - RunPod Docker image (CUDA 12.1, Applio, Demucs, RMVPE)
- `app.py` - Desktop app (PyWebView wrapper)
- `build_exe.py` - PyInstaller EXE build script
- `CLAUDE.md` - Project context for Claude Code
- `HANDOFF.md` - Architecture decisions from web Claude conversation

## Workflow Rules
- **항상 코드 수정 후 git commit + git push까지 완료할 것**
- **코드 변경 후 반드시 pytest 실행하여 회귀 확인**

## Important Conventions
- RunPod client variable: `runpod_client` (not `runpod` - conflicts with module)
- Korean UI, mixed Korean/English comments
- Dark theme: #0a0a14 bg, #8b5cf6 violet accent
- No frameworks - pure HTML/CSS/JS frontend
- Pause/resume 기능 제거됨 (HTTP 410 Gone 반환)

## Infrastructure (already set up)
- GitHub Actions: auto Docker build on push (Dockerfile/runpod_handler.py changes)
- Docker registry: ghcr.io/sonjeongwons/ai_record_studio
- RunPod endpoint: configured and ready
- GitHub repo: sonjeongwons/ai_record_studio

## Current Status (2026-04-01)
- v35 모델 학습 완료, KLM49_HFG + RIN_E3 이중 pretrained 지원
- 한국어 노래 (KLM49) + 영어 팝송 (RIN_E3) 모두 변환 가능 — UI에서 선택
- PC 간 클라우드 동기화 (R2 백업/복원) 구현 완료
- 교차 검증 완료: API↔프론트엔드 100% 일치, handler↔server 데이터 계약 정상
- voice_models DB에 pretrained_model 컬럼 추가 (어떤 pretrained로 학습했는지 기록)

## Training Parameters (v35)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송)
- Epochs: 150, Batch: 4, Sample rate: 40kHz
- F0: RMVPE, Embedder: ContentVec, index_rate: 0.35
- Overtraining detector: 50 epoch threshold

## Memories
- [feedback_no_unnecessary_questions.md](feedback_no_unnecessary_questions.md) - 이미 설정된 인프라에 대해 불필요한 질문 금지
- [feedback_harness_engineering.md](feedback_harness_engineering.md) - 하네스 엔지니어링 방식으로 프로젝트 운영, 에이전트 자율 협업
- [project_v13_pipeline_fixes.md](project_v13_pipeline_fixes.md) - v29 기계음 원인 진단 및 v13 수정 내역
- [project_v23_hpss_revert.md](project_v23_hpss_revert.md) - v22 HPSS 파괴 확인, v23 리버트, 향후 개선 로드맵
