# AI Voice Studio - Claude Code 핸드오프 문서
# ═══════════════════════════════════════════════════════
# 이 문서는 claude.ai에서의 전체 대화 내용을 요약한 것입니다.
# Claude Code가 이 문서를 읽고 프로젝트를 이어서 진행할 수 있습니다.
# 생성일: 2026-02-19
# ═══════════════════════════════════════════════════════


## 1. 프로젝트 개요

### 의뢰인 상황
- 장홍권: 전직 가수, 목 부상으로 더 이상 노래 불가
- 목표: 기존 녹음물로 AI 보이스 모델을 학습시켜 새 앨범 제작
- 의뢰자(개발자)가 이 기술을 상용 제품으로 만들어 다른 고객에게도 판매 예정

### 제품 형태
- 클라이언트: 로컬에서 실행되는 Python FastAPI 서버 + 웹 브라우저 UI
- GPU 처리: RunPod Serverless API (클라우드)
- 라이선스/과금 시스템: **나중에 추가 예정** (현재 단계에서는 제외)
- 현재 목표: **파일 업로드 → 학습 → 음원 변환**이 실제로 동작하는 MVP


## 2. 핵심 기술 결정사항 (확정)

### 2-1. 보컬 AI 모델: RVC v2 (Applio) ✅

**SVC(Singing Voice Conversion)가 필요함 — TTS가 아님!**
- 입력: 오디오(가이드 보컬) → 출력: 변환된 노래 목소리
- 멜로디, 리듬, 감정, 비브라토를 원본에서 보존
- TTS 모델(Fish Speech, CosyVoice, IndexTTS, Qwen3-TTS 등)은 **모두 부적합** (텍스트→음성이므로)

**RVC v2를 선택한 이유:**
- So-VITS-SVC는 2023년 아카이브됨, 유지보수 중단
- RVC는 4GB VRAM만 필요 (So-VITS는 10GB)
- 2-3배 빠른 학습
- HuBERT 임베딩 인덱스 기반 검색으로 음색 보존 우수
- Applio (MIT 라이선스): 활발히 유지보수 중, Docker 지원

**RVC 학습 파라미터:**
- F0 추출: RMVPE (가장 정확한 피치)
- 임베더: ContentVec (기본) 또는 Spin (발음 더 좋음)
- 사전학습 모델: 항상 사용 (학습 시간 대폭 단축)
- 에폭: 300-500 (데이터 양에 따라)
- 배치 사이즈: 8 (RTX 4090 기준)
- 저장 빈도: 50 에폭마다
- 학습 데이터: 10-60분 클린 보컬
- 품질 목표: MOS >4.0, 화자 유사도 >85%

### 2-2. GPU 클라우드: RunPod ✅

**RunPod을 선택한 이유:**
| 항목 | RunPod | Replicate | Modal | Fal AI |
|------|--------|-----------|-------|--------|
| 커스텀 Docker | ✅ 완전 지원 | ⚠️ Cog만 | ✅ Python | ⚠️ 엔터프라이즈 |
| RTX 4090 가격 | $0.34/hr | N/A | $0.55/hr | 문의 |
| A100 가격 | $1.64/hr | $8.28/hr | $2.78/hr | $1.89/hr |
| 장시간 학습 | ✅ Pod 모드 | ❌ 타임아웃 | ✅ 가능 | ❌ 불가 |
| 콜드 스타트 | ~2s | 느림 | 2-4s | 빠름 |

**RunPod 사용 방식:**
- Pod 모드: 장시간 학습 (3-6시간)
- Serverless 모드: 추론/변환 (30초 이내)
- GPU: RTX 4090 Community Cloud ($0.34/hr) — RVC는 24GB VRAM이면 충분
- FlashBoot: <200ms 콜드 스타트

### 2-3. 학습 데이터 구성

**사용 가능한 데이터 유형:**
1. 솔로 보컬 (MR 제거) — ⭐⭐⭐⭐⭐ 최고
2. OST/앨범 곡 (Demucs로 보컬 분리) — ⭐⭐⭐⭐
3. 라이브 공연 — ⭐⭐⭐
4. 인터뷰 영상 — ⭐⭐⭐ (음색 풍부도, 발음 명료도 향상)
5. SNS 라이브 — ⭐⭐~⭐⭐⭐

**비율 권장:** 노래 60-70% + 말하기 30-40%
**최소 기준:** 노래가 50% 이상이어야 함

**인터뷰/영상 데이터 전처리:**
- FFmpeg으로 오디오 추출
- Demucs/UVR5로 BGM 제거
- pyannote/speaker-diarization-3.1로 화자 분리 (홍권 님 목소리만)
- RNNoise로 노이즈 제거
- 5-15초 세그먼트로 분할


## 3. 비용 분석

### 학습 비용 (1시간 오디오, RunPod RTX 4090 Community)
| 시나리오 | GPU 시간 | 비용 |
|---------|---------|------|
| 300 에폭 (권장) | ~5.3h | $1.80 (≈₩2,400) |
| 500 에폭 (고품질) | ~8.8h | $2.99 (≈₩4,000) |
| 200 에폭 (테스트) | ~3.8h | $1.29 (≈₩1,700) |

### 전체 프로젝트 비용 (10곡 앨범)
| 작업 | GPU 시간 | 비용 |
|------|---------|------|
| 1차 학습 (300에폭) | ~5.3h | $1.80 |
| 테스트 변환 5곡 | ~0.04h | $0.01 |
| 재학습 (튜닝 후) | ~5.3h | $1.80 |
| 최종 변환 10곡 | ~0.08h | $0.03 |
| 이펙트 처리 | ~0.05h | $0.02 |
| **합계** | **~10.8h** | **$3.66 (≈₩4,900)** |


## 4. 현재 구현 상태

### 4-1. 완성된 파일 (voice-studio/)

```
voice-studio/
├── server.py          # ✅ FastAPI 백엔드 (모든 API 엔드포인트)
├── static/
│   └── index.html     # ✅ 웹 UI (다크 테마, 4개 탭)
├── requirements.txt   # ✅ Python 의존성
├── start.bat          # ✅ Windows 실행 스크립트
├── start.sh           # ✅ Mac/Linux 실행 스크립트
├── README.md          # ✅ 사용법 문서
├── uploads/           # 업로드 파일 저장소
├── models/            # 학습된 모델 저장소
└── output/            # 변환 결과 저장소
```

### 4-2. server.py 주요 구조

```python
# SQLite 테이블:
# - training_files: 업로드된 파일 관리
# - voice_models: 학습된 AI 모델 (.pth, .index)
# - conversions: 변환 이력
# - jobs: 비동기 작업 상태 추적

# API 엔드포인트:
# GET  /                    → 웹 UI
# GET  /api/config          → RunPod 설정 조회
# POST /api/config          → RunPod 설정 저장
# POST /api/upload          → 파일 업로드 (다중)
# GET  /api/files           → 업로드 파일 목록
# DELETE /api/files/{id}    → 파일 삭제
# POST /api/train           → 학습 시작 (RunPod 비동기)
# POST /api/convert         → 변환 시작 (RunPod 비동기)
# GET  /api/jobs/{id}       → 작업 상태 조회
# GET  /api/jobs            → 전체 작업 목록
# GET  /api/models          → 모델 목록
# DELETE /api/models/{id}   → 모델 삭제
# GET  /api/conversions     → 변환 이력
# GET  /api/download/{name} → 변환 파일 다운로드
# GET  /api/stats           → 대시보드 통계

# RunPod 클라이언트:
# - submit_job(): 비동기 작업 제출
# - check_status(): 상태 폴링
# - encode_files(): base64 인코딩
# - poll_runpod_job(): 백그라운드 스레드에서 완료까지 폴링
```

### 4-3. index.html 웹 UI 구조

```
4개 탭:
1. 홈 (대시보드)     → 통계, 최근 작업
2. 학습              → 파일 드래그&드롭 업로드 → 설정 → 학습 → 프로그레스 링
3. 변환              → 모델 선택 → 음원 업로드 → 피치/인덱스 조절 → 변환 → 다운로드
4. 설정              → RunPod API Key, Endpoint ID

디자인: 다크 테마, #0a0a14 배경, 바이올렛 포인트(#8b5cf6)
```

### 4-4. 동작 테스트 결과
```
GET /: 200 ✅
GET /api/stats: {'files': 0, 'models': 0, 'conversions': 0} ✅
GET /api/config: {'is_configured': False} ✅
GET /api/models: {'models': []} ✅
POST /api/upload: {'files': [...], 'count': 1} ✅
```
**로컬 서버 + 웹 UI + SQLite + API 모두 정상 동작 확인됨**


## 5. 아직 구현되지 않은 것들 (TODO)

### 5-1. 🔴 필수 — RunPod Docker 이미지 (가장 중요!)
현재 클라이언트는 만들어졌지만, **RunPod에서 실행될 서버사이드 핸들러**가 없음.
이것이 실제로 학습/변환이 동작하려면 반드시 필요.

```python
# runpod_handler.py — RunPod Serverless에서 실행됨
# Docker 이미지에 포함해야 할 것:
# - Applio/RVC v2 (학습 + 추론)
# - Demucs (보컬 분리)
# - FFmpeg (오디오/영상 처리)
# - pyannote (화자 분리)
# - RNNoise (노이즈 제거)

import runpod
import base64

def handler(job):
    input = job["input"]
    task_type = input["task_type"]
    
    if task_type == "preprocess":
        # 1. 영상이면 FFmpeg으로 오디오 추출
        # 2. Demucs로 보컬 분리
        # 3. pyannote로 화자 분리
        # 4. RNNoise로 노이즈 제거
        # 5. 세그먼트 분할 (5-15초)
        return {"segment_count": N, "total_duration": seconds}
    
    elif task_type == "train":
        # 1. 오디오 파일 디코딩
        # 2. RVC v2 전처리 (HuBERT 임베딩, F0 추출)
        # 3. 학습 실행 (300-500 에폭)
        # 4. FAISS 인덱스 생성
        # 5. .pth + .index base64 인코딩하여 반환
        return {
            "pth_data": base64_pth,
            "index_data": base64_index,
            "pth_filename": "model.pth",
            "index_filename": "model.index",
            "model_name": name,
            "epochs_trained": N,
            "training_time_seconds": T
        }
    
    elif task_type == "convert":
        # 1. 모델 파일 디코딩 (.pth, .index)
        # 2. 입력 오디오 디코딩
        # 3. RVC v2 추론 (변환)
        # 4. 결과 오디오 base64 인코딩
        return {
            "converted_audio": base64_audio,
            "filename": "converted_song.wav",
            "processing_time_seconds": T
        }

runpod.serverless.start({"handler": handler})
```

**Docker 이미지 구성:**
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# Applio/RVC v2 설치
# Demucs, pyannote, FFmpeg 설치
# runpod_handler.py 복사
# ENTRYPOINT: python runpod_handler.py
```

### 5-2. 🟡 중요 — 전처리 파이프라인 통합
현재 server.py에는 전처리(preprocess) API 경로가 없음.
학습 전에 자동으로 전처리가 실행되도록 해야 함:
- 영상 파일 → 오디오 추출
- 보컬 분리 (Demucs)
- 화자 분리 (pyannote)
- 노이즈 제거
- 세그먼트 분할

### 5-3. 🟡 중요 — 에러 핸들링 강화
- 네트워크 실패 시 재시도 (exponential backoff)
- 학습 중간 체크포인트 복구
- 대용량 파일 업로드 시 청크 전송
- RunPod 잔액 부족 시 안내

### 5-4. 🟢 나중에 — 라이선스/과금 시스템
**의뢰인이 "나중에"라고 명시함**
- 라이선스 서버 (FastAPI + PostgreSQL)
- 고객별 라이선스 키 발급/만료
- GPU 크레딧 관리
- 사용량 통계

### 5-5. 🟢 나중에 — 고급 기능
- 오디오 이펙트 (리버브, 에코, 오토튠, EQ, 컴프레서, 디에서)
- A/B 비교 플레이어
- 배치 변환 (앨범 전체)
- PyInstaller exe 패키징 (~150-200MB)
- 자동 업데이트


## 6. 상용화 수익 모델 (참고)

| 항목 | 고객 청구 | 실제 비용 | 마진 |
|------|----------|----------|------|
| 초기 세팅 (프로그램 + 첫 학습) | ₩3,000,000 | ~₩5,000 | 99.8% |
| 추가 학습 (1회) | ₩300,000-500,000 | ~₩2,000 | 99%+ |
| 월 유지보수 | ₩100,000-200,000 | ~₩0 | 100% |
| GPU 크레딧 (곡당) | ₩10,000-20,000 | ~₩15 | 99%+ |


## 7. Claude Code에게 전달 사항

### 즉시 진행해야 할 작업 우선순위:
1. **RunPod Serverless Handler + Dockerfile** 작성 (RVC v2/Applio 기반)
2. **전처리 파이프라인** server.py에 통합
3. **엔드투엔드 테스트** — 실제 오디오 파일로 학습→변환 동작 확인
4. 에러 핸들링 강화

### 코드 컨벤션:
- Python 3.10+
- FastAPI + SQLite
- 프론트엔드: 순수 HTML/CSS/JS (프레임워크 없음)
- 한국어 UI, 주석은 한국어/영어 혼용
- 다크 테마 디자인 유지

### 개발자 참고사항:
- Applio GitHub: https://github.com/IAHispano/Applio
- Applio Docker: 공식 Docker 이미지 제공
- RunPod Serverless: https://docs.runpod.io/serverless
- RunPod Python SDK: `pip install runpod`
- 모든 GPU 작업은 RunPod에서 처리, 클라이언트는 CPU만 사용
