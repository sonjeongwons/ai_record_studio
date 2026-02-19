@echo off
chcp 65001 >nul
title AI Voice Studio

echo.
echo  ╔══════════════════════════════════╗
echo  ║     AI Voice Studio v1.0        ║
echo  ╚══════════════════════════════════╝
echo.

:: Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org 에서 Python 3.10+ 를 설치해주세요.
    pause
    exit /b
)

:: 패키지 설치 (최초 1회)
if not exist ".venv" (
    echo [설치] 가상환경 생성 중...
    python -m venv .venv
    call .venv\Scripts\activate
    echo [설치] 패키지 설치 중...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

:: 폴더 생성
if not exist "uploads" mkdir uploads
if not exist "models" mkdir models
if not exist "output" mkdir output

echo.
echo  서버 시작 중... 브라우저가 자동으로 열립니다.
echo  종료하려면 이 창을 닫으세요.
echo.

python server.py
pause
