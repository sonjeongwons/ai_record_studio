---
name: code-audit-2026-04
description: 2026-04-01 전체 코드 감사 결과 — 보안/버그/테스트 수정 내역 + 잔여 이슈
type: project
---

## 2026-04-01 전체 코드 감사 결과

4개 에이전트 병렬 배치하여 server.py, index.html, runpod_handler.py, Dockerfile, tests 전체 감사 수행.

### 수정 완료 (commit 0770a64)

**보안:**
- runpod_handler.py line 1065: path traversal 수정 (Path().name 추가)
- index.html: 서버 응답 toast에 escapeHtml 적용 (6곳)
- runpod_handler.py: 모델 파일 크기 검증 추가 (1MB 미만 거부)

**버그:**
- server.py: SQLite 백업 리소스 누수 (try/finally 적용)
- server.py: 업로드 중복 체크 race condition (단일 트랜잭션으로 통합)
- server.py: import json as _json 루프 내 반복 → 모듈 레벨 json + JSONDecodeError
- index.html: switchResultAudio null 체크, JSON.parse 빈 catch 수정

**코드 정리:**
- runpod_handler.py: 비활성 pitch pre/post-shift 데드 코드 40줄 제거
- runpod_handler.py: locals() 기반 GPU cleanup → 명시적 변수 초기화
- tests/conftest.py: 미사용 tmp_data_dir fixture + tempfile import 제거

**테스트 (34→43):**
- +9 신규: download (3), cancel, save-to-folder, preprocess-reset (2), chunk-upload, path-traversal
- 기존 약한 assertion 강화: 200|404 → 정확한 상태 코드
- config 테스트: 마스킹된 API 키 값 검증 추가

### 잔여 이슈 (수정 미완)

1. **CVE-2025-32434 (HIGH)**: PyTorch 2.1.0의 torch.load() RCE 취약점. 2.6.0+ 업그레이드 필요하나 Applio/fairseq 호환성 미검증.
2. **FastAPI DeprecationWarning**: on_event → lifespan 마이그레이션 (기능 영향 없음)
3. **poll_runpod_job 들여쓰기**: while True가 1-space indent (동작 정상, 순수 코스메틱)
4. **CORS allow_origins=["*"]**: 데스크톱 앱이므로 현실적 위험 낮음, 추후 제한 가능

### 에이전트 리서치 결과 (RVC 최신 동향)
- Applio 3.6.2 (유지보수 모드) — Python 3.12 지원. 우리는 3.10에서 안정 유지.
- KLM49_HFG는 최신 한국어 pretrained — 교체 불필요
- RMVPE는 여전히 최고 F0 방법. FCPE는 추론 시 대안으로 테스트 가치 있음.
- 학습 파라미터 (150에폭, batch 4, 40kHz)는 커뮤니티 권장과 일치

**Why:** 정기적 코드 감사로 기술 부채와 보안 위험을 조기 발견/관리
**How to apply:** 다음 감사 시 잔여 이슈 우선 처리. PyTorch 업그레이드는 별도 브랜치에서 테스트.
