# AI Voice Studio

AI 보이스 클로닝 스튜디오 — RVC v2 기반 Singing Voice Conversion

## 빠른 시작

### 고객용 (.exe)
`AI Voice Studio.exe` 더블클릭 — Python 설치 불필요

### 개발자용
```bash
pip install -r requirements.txt
python app.py        # 데스크톱 앱 (PyWebView)
python server.py     # 개발 서버 (브라우저)
```

## 사용법

### 1단계: RunPod 설정
- [runpod.io](https://runpod.io) 가입
- Serverless Endpoint 배포 (RVC Docker 이미지)
- 설정 페이지에서 API Key + Endpoint ID 입력
- (선택) Cloudflare R2 설정 (대용량 파일 전송 + PC 간 동기화)

### 2단계: 학습
- 학습 탭에서 음원/인터뷰/라이브 영상 파일 업로드
- 사전학습 모델 선택: KLM49 (한국어) 또는 RIN_E3 (영어 팝/다국어)
- 모델 이름, 에폭 수 설정 후 "학습 시작"
- 약 30분~2시간 소요 (RunPod RTX 4090, 150 에폭 기준)

### 3단계: 변환
- 변환 탭에서 학습된 모델 선택
- 가이드 보컬(노래) 파일 업로드
- 피치/인덱스 비율 조절 후 "변환 시작"
- 완료 후 다운로드 (보컬 + 최종 믹스)

## 지원 파일 형식
- 오디오: WAV, MP3, FLAC, OGG, M4A
- 영상: MP4, MKV, WEBM (자동 오디오 추출)

## 테스트
```bash
python -m pytest tests/ -v   # 34개 API 테스트
```

## EXE 빌드 (고객 배포용)
```bash
python build_exe.py
# → dist/AI Voice Studio/AI Voice Studio.exe
```

## 구조
```
ai_record_studio/
├── app.py               # 데스크톱 앱 (PyWebView + 내장 서버)
├── build_exe.py          # PyInstaller EXE 빌드
├── server.py             # FastAPI 백엔드 (35+ API)
├── static/index.html     # 웹 UI (다크 테마, 4탭)
├── runpod_handler.py     # RunPod GPU 핸들러 (전처리/학습/변환)
├── Dockerfile            # RunPod Docker 이미지
├── requirements.txt      # Python 의존성
├── tests/                # API 테스트 (34개)
├── uploads/              # 업로드된 파일
├── models/               # 학습된 모델
├── output/               # 변환된 음원
└── preprocessed/         # 전처리 세그먼트
```

## 예상 비용 (RunPod RTX 4090)
- 학습 1회 (150에폭, 25분 데이터): ~$0.50~$1.50
- 변환 1곡: ~$0.01~$0.05
- 전처리 1곡: ~$0.02~$0.05
