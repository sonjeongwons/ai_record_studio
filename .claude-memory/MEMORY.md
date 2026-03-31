# AI Voice Studio - Project Memory

## Project Overview
- AI Voice Cloning Studio for singer 장홍권
- Tech: RVC v2 (Applio) + RunPod Serverless + FastAPI + vanilla HTML/CSS/JS
- Location: `c:\ai_record_studio\voice-studio\`

## Key Files
- `server.py` - FastAPI backend (all API endpoints, RunPod client with retry)
- `static/index.html` - Web UI (dark theme, 4 tabs, toast notifications, audio player)
- `runpod_handler.py` - RunPod Serverless GPU handler (preprocess/train/convert)
- `Dockerfile` - RunPod Docker image (CUDA 12.1, Applio, Demucs, RMVPE)
- `CLAUDE.md` - Project context for Claude Code
- `HANDOFF.md` - Architecture decisions from web Claude conversation

## Workflow Rules
- **항상 코드 수정 후 git commit + git push까지 완료할 것**

## Important Conventions
- RunPod client variable: `runpod_client` (not `runpod` - conflicts with module)
- Korean UI, mixed Korean/English comments
- Dark theme: #0a0a14 bg, #8b5cf6 violet accent
- No frameworks - pure HTML/CSS/JS frontend
- Windows dev environment - use UTF-8 encoding for file reads

## Infrastructure (already set up)
- GitHub Actions: auto Docker build on push (Dockerfile/runpod_handler.py changes)
- Docker registry: ghcr.io/sonjeongwons/ai_record_studio
- RunPod endpoint: configured and ready
- GitHub repo: sonjeongwons/ai_record_studio

## Current Status (2026-03-23)
- v30 모델 학습 완료, v22(HPSS) 적용으로 v18/v12가 치명적 품질 저하
- v23 커밋 (9e402ee + 29df6b0): HPSS 비활성화 + index_rate 0.50 상향
- Docker 자동 빌드 트리거됨, 빌드 완료 후 재변환 예정
- 다음: v23으로 재변환 (v17 수준 복원 확인) → 모델 재학습(v31, epoch 단축) 검토

## Pause/Resume Feature (added 2026-03-04)
- `server.py`: `RunPodClient.cancel_runpod_job()`, `pause_job`, `resume_job` endpoints
- `server.py`: `_active_job_states` dict for resume params; `pause_state_json` DB column
- `server.py`: `_is_job_cancelled()` checks `("failed", "cancelled", "paused")`
- `index.html`: CSS `.btn-pause`, `.btn-resume`, `.job-paused-area`, `.badge-paused`
- `index.html`: Pause buttons + paused panels for preprocess/train/convert
- `index.html`: State vars `_preprocessPaused`, `_trainPaused`, `_convertPaused`, `_preprocessPollErrors`
- Poll functions guard on `_xxxPaused`, handle `paused`/`cancelled`, 20-error auto-pause
- Cancel endpoints call RunPod cancel API to stop billing

## Memories
- [feedback_no_unnecessary_questions.md](feedback_no_unnecessary_questions.md) - 이미 설정된 인프라에 대해 불필요한 질문 금지
- [feedback_harness_engineering.md](feedback_harness_engineering.md) - 하네스 엔지니어링 방식으로 프로젝트 운영, 에이전트 자율 협업
- [project_v13_pipeline_fixes.md](project_v13_pipeline_fixes.md) - v29 기계음 원인 진단 및 v13 수정 내역
- [project_v23_hpss_revert.md](project_v23_hpss_revert.md) - v22 HPSS 파괴 확인, v23 리버트, 향후 개선 로드맵
