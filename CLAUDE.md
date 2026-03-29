# CLAUDE.md — AI Voice Studio

## 프로젝트 컨텍스트
이 프로젝트는 AI 보이스 클로닝 스튜디오입니다.
전직 가수(장홍권)의 기존 녹음물로 AI 보이스 모델을 학습시켜 새 앨범을 제작하는 것이 목표이며,
이후 상용 제품으로 다른 고객에게도 판매할 예정입니다.

**`HANDOFF.md`에 전체 아키텍처 결정, 모델 선택 이유, 비용 분석이 상세히 정리되어 있습니다.**

## 기술 스택
- 보컬 AI: RVC v2 (Applio) — SVC(Singing Voice Conversion), TTS 아님!
- GPU 클라우드: RunPod Serverless (RTX 4090)
- 백엔드: Python FastAPI + SQLite
- 프론트엔드: 순수 HTML/CSS/JS (프레임워크 없음)
- 데스크톱 앱: PyWebView + PyInstaller (.exe 패키징)
- 배포: 단일 .exe (고객 Python 설치 불필요) + RunPod Docker 이미지
- 로깅: Python `logging` 모듈 (콘솔 + 파일 server.log, RotatingFileHandler 5MB x 3)
- 테스트: pytest + FastAPI TestClient (격리된 임시 DB 환경)
- CI/CD: GitHub Actions (Docker 빌드 + pytest 자동 실행)

## 현재 상태 (2026-03-29 업데이트)
- ✅ 클라이언트 (server.py + index.html) 완성, API 테스트 18/18 통과
- ✅ RunPod Serverless Handler (runpod_handler.py) 구현 완료
- ✅ 네트워크 장애 복원력 강화 (서버 재시작/네트워크 끊김 시 자동 복구)
- ✅ 구조화 로깅 (print → logging 전환)
- ✅ pytest 테스트 프레임워크 + CI/CD 테스트 파이프라인
- ✅ 장함수 리팩토링 (헬퍼 함수 추출)
- ✅ 에러 처리 개선 (silent except에 debug 로깅 추가)

## 컨벤션
- Python 3.10+, 타입힌트 사용
- 주석: 한국어/영어 혼용
- UI: 한국어, 다크 테마 (#0a0a14 배경, #8b5cf6 바이올렛 포인트)
- 라이선스/과금 시스템은 나중에 추가 (현재 제외)
- RunPod 클라이언트 변수명: `runpod_client` (runpod 모듈과 구분)
- **코드 수정 후 반드시 git commit + git push까지 완료**
- **코드 변경 후 반드시 `python3 -m pytest tests/ -v` 실행하여 회귀 확인**

## 하네스 엔지니어링 제약
- 모든 API 변경은 테스트 추가/수정 필수 (`tests/test_api.py`)
- silent `except Exception: pass` 금지 — 최소 `logger.debug()` 포함
- 새 print() 사용 금지 — `logger.info/warning/error/critical` 사용
- 장함수(100줄+) 추가 금지 — 헬퍼 함수로 분리

## 주요 명령어
```bash
# 개발 모드 (서버 직접 실행)
python server.py

# 테스트 실행
python3 -m pytest tests/ -v

# 데스크톱 앱 테스트 (pywebview 윈도우)
python app.py

# EXE 빌드 (고객 배포용)
pip install -r requirements.txt
python build_exe.py
# → dist/AI Voice Studio/AI Voice Studio.exe

# Docker 빌드 (RunPod GPU용)
docker build -t ai-voice-studio:latest .
```

## 파일 구조
```
voice-studio/
├── CLAUDE.md            ← 이 파일 (하네스 제약 문서)
├── HANDOFF.md           ← 전체 아키텍처/결정사항 상세 문서
├── app.py               ← 데스크톱 앱 (PyWebView 윈도우 + 내장 서버)
├── build_exe.py         ← PyInstaller EXE 빌드 스크립트
├── server.py            ← FastAPI 백엔드 (듀얼 패스: .exe/개발 모드)
├── static/index.html    ← 웹 UI (토스트, 오디오 플레이어, 배치 변환, 전처리)
├── runpod_handler.py    ← RunPod Serverless GPU 핸들러
├── Dockerfile           ← RunPod Docker 이미지 (Applio/Demucs/RMVPE 포함)
├── .dockerignore        ← Docker 빌드 제외 목록
├── requirements.txt     ← Python 의존성
├── pytest.ini           ← pytest 설정
├── tests/               ← 테스트 코드
│   ├── conftest.py      ← 격리 환경 fixture
│   └── test_api.py      ← API 엔드포인트 테스트
├── .github/workflows/
│   ├── docker-build.yml ← RunPod Docker 자동 빌드
│   └── test.yml         ← pytest 자동 실행
├── uploads/             ← 업로드 파일
├── models/              ← 학습된 모델
├── output/              ← 변환 결과
└── chunks/              ← 청크 업로드 임시
```

## API 엔드포인트 (전체)
```
GET  /                          → 웹 UI
GET  /api/config                → RunPod 설정 조회
POST /api/config                → RunPod 설정 저장
POST /api/upload                → 파일 업로드 (다중)
POST /api/upload/chunk          → 청크 업로드 (대용량)
GET  /api/files                 → 업로드 파일 목록
DELETE /api/files/{id}          → 파일 삭제
POST /api/preprocess            → 전처리 시작 (RunPod)
GET  /api/preprocess/status     → 전처리 상태 조회
POST /api/train                 → 학습 시작 (RunPod)
POST /api/convert               → 변환 시작 (RunPod)
GET  /api/jobs/{id}             → 작업 상태 조회
GET  /api/jobs/active           → 현재 실행 중 작업
GET  /api/jobs                  → 전체 작업 목록
POST /api/jobs/cleanup          → 작업 정리
POST /api/jobs/{id}/cancel      → 작업 취소
POST /api/jobs/{id}/pause       → 작업 일시정지
POST /api/jobs/{id}/resume      → 작업 재개
GET  /api/models                → 모델 목록
DELETE /api/models/{id}         → 모델 삭제
POST /api/models/{id}/rename    → 모델 이름 변경
POST /api/models/{id}/quality   → 품질 점수 업데이트
GET  /api/conversions           → 변환 이력
DELETE /api/conversions/{id}    → 변환 이력 삭제
GET  /api/download/{name}       → 파일 다운로드
GET  /api/stats                 → 대시보드 통계
GET  /api/health                → 서버/RunPod/DB 상태
GET  /api/system-info           → 시스템 정보
```
