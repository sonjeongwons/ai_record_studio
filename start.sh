#!/bin/bash
echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║     AI Voice Studio v1.0        ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# 가상환경 설치
if [ ! -d ".venv" ]; then
    echo "[설치] 가상환경 생성 중..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[설치] 패키지 설치 중..."
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

mkdir -p uploads models output

echo ""
echo "  서버 시작 중... http://localhost:8000"
echo "  종료: Ctrl+C"
echo ""

python server.py
