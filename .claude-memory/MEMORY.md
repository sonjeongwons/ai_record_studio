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

## Current Status (2026-04-03)
- **v36 모델 "My Voice v36" 학습 완료** (소스 9개 재전처리 + v36 코드로 학습)
- v35 모델 R2 클라우드 업로드 완료 (index 357MB + pth 53MB)
- R2 전체: 9,207개 파일, 28.6GB (모델/변환/전처리/학습 데이터 모두 포함)
- **v36 음질 최적화 전체 적용 완료**:
  - rms=0, index 0.40, protect 0.35, FCPE F0 추가
  - BS-Roformer SOTA 분리 (전처리+변환, 모델 가중치 Dockerfile 포함)
  - 2-pass loudnorm -14 LUFS, 원본 보컬 10% 블렌딩
  - 48kHz SR 보존, 숨소리 보존 강화, 에폭 체크포인트 비교 지원
- Seed-VC 평가 완료: RVC 유지 결정 (Seed-VC DNSMOS↓, 아카이브됨)
- YingMusic-SVC: 향후 주목할 차세대 SVC (코드 공개 진행 중)
- KLM49_HFG (한국어 노래) + RIN_E3 (영어 팝/다국어) 이중 pretrained 지원
- PC 간 클라우드 동기화 (Cloudflare R2 백업/복원) 구현 완료
  - R2 백업/복원 시 **중복 파일 자동 스킵** (key+size 비교, 변경분만 전송)
- 테스트 43/43 통과 (pytest)
- ruff + bandit 정적 분석 클린
- **CVE-2025-32434**: PyTorch 2.1.0 RCE 취약점 인지 — 2.6.0+ 업그레이드 예정

## Training Parameters (v35 — 한국어 커뮤니티 최적값)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: 150, Batch: 4, Sample rate: 40kHz
- F0: RMVPE, Embedder: ContentVec (768-dim)
- index_rate: 0.40, rms_mix_rate: 0.0, protect: 0.35, filter_radius: 3
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
- v31 48kHz 시도 → 보코더 기계음 악화 → v32에서 40kHz 복원 (RVC Issue #119/#514)
- v33~v34 후처리 최적화: 과도한 고역 EQ 커팅 제거, 노이즈 게이트 추가, asoftclip 제거
- v35 KLM49_HFG 도입 + Demucs 보컬분리 전처리 추가 (근본 원인 해결)

## 음질 개선 히스토리 (v30~v35)
- **근본 원인 발견 (2026-03-31)**: 학습 소스 9개 중 6개에 MR(반주) 포함
  → 모델이 베이스/드럼/기타를 "목소리의 일부"로 학습 = 기계음 80%의 원인
- v30: 기본 pretrained, 800ep → 심각한 오버트레이닝 + MR 오염
- v31: 48kHz 시도 → 보코더 기계음 악화
- v32: 40kHz 복원, 300ep → MR 오염 미해결
- v33 (KLM49): 한국어 pretrained, 150ep, batch4 → 약간 개선, MR 미해결
- v35: 전처리에 Demucs 보컬분리 추가 → MR 자동 제거 → 학습 완료 (My Voice v35)
- **v36 (현재)**: BS-Roformer SOTA 분리, rms=0, loudnorm -14 LUFS, 원본 블렌딩 → **학습 완료 (My Voice v36)**
- **다음: v36 모델로 3곡 변환 (기본값 프리셋)**

## 타겟 곡 3개
- "01_Breaking Through (4824 Wave).wav" — 영어, 팝/록, 48kHz/24bit WAV
- "플레이브 - 기다릴게.mp3" — 한국어, K-POP, 44.1kHz MP3
- "Jeremy Zucker - comethru ft. Bea Miller.mp3" — 영어, 인디팝, 44.1kHz MP3

## 학습 소스 음원 9개 (mp3record/ 폴더)
- 07a51a19, 5a8a4c9a, a2eebb1e, 90d37ee7, 79dde5c2 — MR 포함 (Demucs 분리 필요)
- 장이정-살다가한번쯤 — MR 포함 (Demucs 분리 필요)
- **장이정-살다가테스트** — ✓ 깨끗한 보컬 (320kbps, 최고 품질)
- **히스토리_tomorrow** — ✓ 깨끗한 보컬 (66초)
- 장이정 학습용 데이터 (1) — 20분, 혼합 (보컬 12분 + MR 8분, Demucs 분리 필요)

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
- [feedback_memory_maintenance.md](feedback_memory_maintenance.md) — 변경 시 CLAUDE.md + 메모리 3종 동시 현행화 필수
