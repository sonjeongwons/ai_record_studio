# CLAUDE.md — AI Voice Studio

## ⚡ 메모리 자동 복원 (필수 — 첫 대화 시 반드시 실행)
> **새 PC이거나 새 대화창이면** `.claude-memory/` 폴더의 모든 .md 파일을 읽어서 맥락을 파악하세요.
> 특히 `MEMORY.md`(종합 현황), `feedback_*.md`(작업 방식), `project_*.md`(기술 이력)를 반드시 확인하세요.
> 이 지시를 건너뛰지 마세요 — 이전 대화에서 축적된 모든 결정사항과 제약이 담겨 있습니다.

## 프로젝트 컨텍스트
이 프로젝트는 AI 보이스 클로닝 스튜디오입니다.
전직 가수(장홍권)의 기존 녹음물(9곡)로 AI 보이스 모델을 학습시켜 새 앨범을 제작하는 것이 목표이며,
이후 상용 제품으로 다른 고객에게도 판매할 예정입니다.

**`HANDOFF.md`에 전체 아키텍처 결정, 모델 선택 이유, 비용 분석이 상세히 정리되어 있습니다.**

## 기술 스택
- 보컬 AI: RVC v2 (Applio) — SVC(Singing Voice Conversion), TTS 아님!
- GPU 클라우드: RunPod Serverless (RTX 4090)
- 백엔드: Python FastAPI + SQLite (WAL 모드)
- 프론트엔드: 순수 HTML/CSS/JS (프레임워크 없음)
- 데스크톱 앱: PyWebView + PyInstaller (.exe 패키징)
- 배포: 단일 .exe (고객 Python 설치 불필요) + RunPod Docker 이미지
- 로깅: Python `logging` 모듈 (콘솔 + 파일 server.log, RotatingFileHandler 5MB x 3)
- 테스트: pytest + FastAPI TestClient (격리된 임시 DB 환경) — 43개
- CI/CD: GitHub Actions (Docker 빌드 + pytest 자동 실행)
- 정적분석: ruff (린터) + bandit (보안)

## 현재 상태 (2026-04-01 업데이트)
- ✅ 클라이언트 (server.py + index.html) 완성, API 테스트 43/43 통과
- ✅ RunPod Serverless Handler (runpod_handler.py) 구현 완료
- ✅ 네트워크 장애 복원력 강화 (서버 재시작/네트워크 끊김 시 자동 복구)
- ✅ 구조화 로깅 (print → logging 전환)
- ✅ pytest 테스트 프레임워크 + CI/CD 테스트 파이프라인
- ✅ 장함수 리팩토링 (헬퍼 함수 추출)
- ✅ 에러 처리 개선 (silent except에 debug 로깅 추가)
- ✅ PC 간 클라우드 동기화 (R2 백업/복원)
- ✅ 배치 변환 + 변환 프리셋 8종
- ✅ 보안 감사 완료 (path traversal, XSS, race condition 수정)
- ⚠️ CVE-2025-32434: PyTorch 2.1.0 RCE — 2.6.0+ 업그레이드 예정 (Applio 호환성 검증 필요)

## 학습 파라미터 (v35 — 한국어 커뮤니티 최적값)
- Pretrained: KLM49_HFG (한국어) / RIN_E3 (다국어/팝송) — UI에서 선택
- Epochs: 150, Batch: 4, Sample rate: 40kHz
- F0: RMVPE, Embedder: ContentVec (768-dim)
- index_rate: 0.35, rms_mix_rate: 0.25, filter_radius: 3
- Overtraining detector: 50 epoch threshold

## 컨벤션
- Python 3.10+, 타입힌트 사용
- 주석: 한국어/영어 혼용
- UI: 한국어, 다크 테마 (#0a0a14 배경, #8b5cf6 바이올렛 포인트, #7a7a90 텍스트)
- 라이선스/과금 시스템은 나중에 추가 (현재 제외)
- RunPod 클라이언트 변수명: `runpod_client` (runpod 모듈과 구분)
- DB 경로는 상대 경로 저장 (이식성), voice_models(name)에 UNIQUE 인덱스
- **코드 수정 후 반드시 git commit + git push까지 완료**
- **코드 변경 후 반드시 `python3 -m pytest tests/ -v` 실행하여 회귀 확인**
- **코드 변경 시 관련 주석/설명/툴팁도 현행화 필수**

## 하네스 엔지니어링 원칙
이 프로젝트는 Martin Fowler의 Harness Engineering 방식으로 운영됩니다.
- **Human On the Loop** — 사용자는 방향(Why), 에이전트가 실행(How) 자율 수행
- **Full Agent Team 모드** — 모든 에이전트팀 풀 가동, 시간/토큰 제한 없이 철저히 진행
- 에이전트끼리 역할 분담: Plan → Code → Test → Review
- 에이전트 간 교차 검증 필수 (자기 검증 X)
- 코드 변경 완료 시 자동으로 git commit + push (매번 사용자 확인 불필요)
- 결과물이 불만족이면 출력물이 아니라 하네스(프로세스/제약)를 개선
- 외부 URL, 검색, 커뮤니티, 공식문서, 도구/라이브러리 자유롭게 사용 가능
- LLM 판단보다 린터, 테스트, 실제 실행 결과를 우선

## 하네스 제약 규칙
- 모든 API 변경은 테스트 추가/수정 필수 (`tests/test_api.py`)
- silent `except Exception: pass` 금지 — 최소 `logger.debug()` 포함
- 새 print() 사용 금지 — `logger.info/warning/error/critical` 사용
- 장함수(100줄+) 추가 금지 — 헬퍼 함수로 분리
- 이미 설정된 인프라에 대해 불필요한 확인 질문 하지 말 것

## 인프라 (이미 설정 완료 — 묻지 말 것)
- GitHub Actions: Docker 자동 빌드 (Dockerfile/runpod_handler.py 변경 시)
- GitHub Actions: pytest 자동 실행 (server.py/tests/ 변경 시)
- Docker registry: ghcr.io/sonjeongwons/ai_record_studio
- RunPod endpoint: 설정 완료 (config.json)
- Cloudflare R2: 대용량 파일 전송 + PC간 동기화
- GitHub repo: sonjeongwons/ai_record_studio

## 프로젝트 이력 (핵심 결정사항)
> 상세 내용은 `.claude-memory/` 디렉토리의 개별 파일 참조.

- **v22 HPSS 실패 → v23 리버트**: HPSS harmonic 추출이 자음/숨소리 제거하여 품질 파괴
- **v13 파이프라인 수정**: asetrate 피치시프트(치명적) 제거, SLICE_DURATION 3.5→5.0s
- **v15 pitch pre-shift 비활성화**: librosa STFT phase vocoder가 formant 미보존 → 기계음 유발
- **v35 KLM49_HFG**: 한국어 노래 최적화 pretrained 도입 (기존 TITAN 교체)
- **2026-04-01 코드 감사**: 보안 9건, 버그 3건 수정. 테스트 34→43. CVE-2025-32434 인지.
- **향후 로드맵**: PyTorch 업그레이드, UVR5 MDX-Net 리드보컬 분리, 원본 보컬 블렌딩

## 메모리 시스템
프로젝트의 모든 대화 맥락과 결정사항은 **두 곳**에 저장됩니다:

1. **`.claude-memory/`** (git에 포함) — 어떤 PC, 어떤 대화창에서든 접근 가능
   - `MEMORY.md` — 종합 현황 인덱스
   - `feedback_*.md` — 작업 방식 피드백
   - `project_*.md` — 기술 결정 이력
2. **`~/.claude/projects/<hash>/memory/`** (Claude Code 내부) — 대화 간 자동 로드

**다른 PC에서 메모리 복원:**
```bash
git pull
setup-claude-memory.bat     # Windows
bash setup-claude-memory.sh  # Mac/Linux
```

## 주요 명령어
```bash
python server.py                    # 개발 서버
python3 -m pytest tests/ -v         # 테스트 (43개)
python app.py                       # 데스크톱 앱
python build_exe.py                 # EXE 빌드
docker build -t ai-voice-studio .   # Docker 빌드
```

## 파일 구조
```
ai_record_studio/
├── CLAUDE.md              ← 이 파일 (하네스 제약 + 메모리 자동복원 지시)
├── HANDOFF.md             ← 전체 아키텍처/결정사항 상세 문서
├── .claude-memory/        ← 대화 맥락/메모리 (git 포함, 크로스 PC 공유)
│   ├── MEMORY.md          ← 종합 인덱스
│   ├── feedback_*.md      ← 작업 방식 피드백
│   └── project_*.md       ← 기술 결정 이력
├── .claude/settings.json  ← Claude Code 프로젝트 설정 (git 포함)
├── setup-claude-memory.*  ← 메모리 복원 스크립트 (bat/sh)
├── server.py              ← FastAPI 백엔드 (~3300줄)
├── static/index.html      ← 웹 UI (~8400줄, 다크 테마, 4탭)
├── runpod_handler.py      ← RunPod GPU 핸들러 (~3500줄)
├── Dockerfile             ← RunPod Docker 이미지
├── app.py                 ← 데스크톱 앱 (PyWebView)
├── build_exe.py           ← PyInstaller EXE 빌드
├── requirements.txt       ← Python 의존성
├── pytest.ini             ← pytest 설정
├── tests/                 ← API 테스트 (43개)
│   ├── conftest.py        ← 격리 환경 fixture
│   └── test_api.py        ← 엔드포인트 테스트
├── .github/workflows/     ← CI/CD
│   ├── docker-build.yml   ← Docker 자동 빌드
│   └── test.yml           ← pytest 자동 실행
└── .gitignore
```

## API 엔드포인트 (전체)
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
