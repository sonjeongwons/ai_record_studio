# CLAUDE.md — AI Voice Studio (전체 맥락 문서)

> **이 파일은 Claude Code가 프로젝트를 열 때 자동으로 읽힙니다.**
> 어떤 PC, 어떤 대화창에서든 `git pull`만 하면 모든 맥락이 복원됩니다.
> 상세 메모리는 `.claude-memory/` 디렉토리에 별도 보관되어 있으며,
> 필요 시 해당 파일들을 직접 Read하여 참조하세요.

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
| 테스트 | pytest + FastAPI TestClient — **43개** |
| CI/CD | GitHub Actions (Docker 빌드 + pytest) |
| 정적분석 | ruff (린터) + bandit (보안) |

## 3. 현재 상태 (2026-04-02)

- ✅ 클라이언트 (server.py + index.html) 완성, API 테스트 **43/43 통과**
- ✅ RunPod Serverless Handler 구현 완료
- ✅ 네트워크 장애 복원력 (서버 재시작/네트워크 끊김 시 자동 복구)
- ✅ PC 간 클라우드 동기화 (Cloudflare R2 백업/복원)
- ✅ 배치 변환 + 변환 프리셋 8종
- ✅ 보안 감사 완료 (path traversal, XSS, race condition 수정)
- ✅ **v36 음질 개선**: rms=0 다이나믹 보존, BS-Roformer SOTA 분리, loudnorm 적용
- ⚠️ **CVE-2025-32434**: PyTorch 2.1.0 RCE — 2.6.0+ 업그레이드 예정

## 4. 변환 파라미터 (v36 — 분석 기반 최적화)

| 파라미터 | 값 | 변경 이유 |
|----------|-----|-----------|
| Pretrained | **KLM49_HFG** (한국어) / **RIN_E3** (다국어) | |
| Epochs | 150, Batch: 4, SR: 40kHz | |
| F0 | RMVPE | |
| **index_rate** | **0.40** (이전 0.35) | 음색 반영 강화 (커뮤니티 0.3-0.5) |
| **rms_mix_rate** | **0.0** (이전 0.25) | 원곡 다이나믹 100% 보존 (기계음 최대 원인) |
| **protect** | **0.35** (이전 0.40) | 자음 보호 + 자연스러운 전환 |
| **vocal_blend** | **10%** (신규) | 원본 보컬 블렌딩으로 숨결감 복원 |
| F0 옵션 | RMVPE + **FCPE** (신규) | FCPE: 풍성한 보컬, 5배 빠름 |
| 보컬 분리 | **BS-Roformer → Demucs** 폴백 | 전처리+변환 모두 SOTA 적용 |
| 후처리 | loudnorm -14 LUFS + presence +1.5dB | 볼륨/명료도 보정 |
| 샘플레이트 | 원본 SR 보존 (48kHz→48kHz) | 다운샘플 방지 |

## 5. 컨벤션

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
> 상세: `.claude-memory/project_v23_hpss_revert.md`

### v13 파이프라인 수정
v29 모델의 기계음 원인 진단: asetrate 피치시프트(치명적), SLICE_DURATION 3.5→5.0s, 고음 오버샘플링 축소, adeclick threshold 조정.
> 상세: `.claude-memory/project_v13_pipeline_fixes.md`

### v15 pitch pre-shift 비활성화
librosa STFT phase vocoder가 formant 미보존 → 이중 적용(pre+post) 시 기계음 유발. 코드 제거 완료.

### v35 KLM49_HFG 도입
한국어 노래 최적화 pretrained (기존 TITAN 교체). RIN_E3는 영어 팝/다국어용으로 병행.

### 2026-04-01 코드 감사
4개 에이전트 병렬 감사: 보안 9건(path traversal, XSS, race condition), 버그 3건 수정. 테스트 34→43. CVE-2025-32434 인지.
> 상세: `.claude-memory/project_code_audit_2026_04.md`

### 2026-04-02 v36 음질 개선 (분석 기반)
6곡 변환 결과 나노 분석: F0 피치 상관계수 0.34-0.73 (목표 0.9+), 다이나믹 레인지 -93dB 압축, 2-4kHz -2.3% 손실.
수정: rms=0, index 0.40, protect 0.35, presence +1.5dB, loudnorm -14 LUFS, BS-Roformer 분리.

### 향후 로드맵
- PyTorch 2.6.0+ 업그레이드 (CVE 대응, Applio 호환성 검증 필요)
- 학습 데이터 보강 (43분→60분+ 권장, BS-Roformer 재전처리)
- 원본 보컬 10-20% 블렌딩 (숨결감 복원)
- 에폭별 체크포인트 수동 비교 (150이 최적인지 검증)
- FastAPI lifespan 마이그레이션

## 10. 주요 명령어

```bash
python server.py                    # 개발 서버
python3 -m pytest tests/ -v         # 테스트 (43개)
python app.py                       # 데스크톱 앱
python build_exe.py                 # EXE 빌드
docker build -t ai-voice-studio .   # Docker 빌드
```

## 11. 파일 구조

```
ai_record_studio/
├── CLAUDE.md              ← 이 파일 (자동 로드, 전체 맥락)
├── HANDOFF.md             ← 아키텍처/결정사항 상세 문서
├── .claude-memory/        ← 상세 메모리 (git 포함)
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
├── tests/                 ← API 테스트 (43개)
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
