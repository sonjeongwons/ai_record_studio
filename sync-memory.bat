@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo  AI Voice Studio - Memory Sync
echo ============================================
echo.

:: 프로젝트 경로
set "PROJECT_DIR=%~dp0"
set "REPO_MEMORY=%PROJECT_DIR%.claude-sync\memory"
set "SESSION_CONTEXT=%PROJECT_DIR%.claude-sync\SESSION_CONTEXT.md"

:: Claude Code 로컬 메모리 경로 자동 감지
:: voice-studio 폴더의 절대 경로에서 프로젝트 해시 생성
set "CLAUDE_BASE=%USERPROFILE%\.claude\projects"

:: 가능한 프로젝트 경로들 (다른 PC에서 다를 수 있음)
set "FOUND_DIR="
for /d %%d in ("%CLAUDE_BASE%\*ai-record-studio*") do (
    if exist "%%d\memory" (
        set "FOUND_DIR=%%d\memory"
        echo [발견] Claude 메모리 경로: %%d\memory
    )
)
for /d %%d in ("%CLAUDE_BASE%\*ai_record_studio*") do (
    if exist "%%d\memory" (
        set "FOUND_DIR=%%d\memory"
        echo [발견] Claude 메모리 경로: %%d\memory
    )
)
for /d %%d in ("%CLAUDE_BASE%\*voice-studio*") do (
    if exist "%%d\memory" (
        set "FOUND_DIR=%%d\memory"
        echo [발견] Claude 메모리 경로: %%d\memory
    )
)

:: 메모리 디렉토리가 없으면 생성 시도
if "%FOUND_DIR%"=="" (
    echo [경고] Claude Code 메모리 경로를 자동 감지하지 못했습니다.
    echo.
    echo 수동 설정: 이 PC에서 Claude Code를 한번 실행한 후 다시 시도하세요.
    echo 또는 아래 경로에 메모리 폴더를 생성합니다...
    echo.

    :: 기본 경로로 생성
    set "FOUND_DIR=%CLAUDE_BASE%\c--ai-record-studio\memory"
    mkdir "%FOUND_DIR%" 2>nul
    echo [생성] %FOUND_DIR%
)

echo.
echo === 1단계: Git Pull (최신 코드 + 메모리 가져오기) ===
cd /d "%PROJECT_DIR%"
git pull
echo.

echo === 2단계: Git 메모리 → Claude 로컬 메모리 복사 ===
if exist "%REPO_MEMORY%" (
    xcopy /Y /Q "%REPO_MEMORY%\*.*" "%FOUND_DIR%\" >nul 2>&1
    echo [완료] %REPO_MEMORY% → %FOUND_DIR%
    echo   복사된 파일:
    for %%f in ("%REPO_MEMORY%\*.*") do echo     %%~nxf
) else (
    echo [경고] Git 메모리 폴더가 없습니다: %REPO_MEMORY%
)
echo.

echo === 3단계: Claude 로컬 메모리 → Git 메모리 동기화 (push용) ===
if exist "%FOUND_DIR%" (
    xcopy /Y /Q "%FOUND_DIR%\*.*" "%REPO_MEMORY%\" >nul 2>&1
    echo [완료] %FOUND_DIR% → %REPO_MEMORY%
)
echo.

echo ============================================
echo  동기화 완료!
echo ============================================
echo.
echo  - CLAUDE.md: 자동 로드됨 (git 내장)
echo  - 메모리 파일: 로컬로 복사됨
echo  - 다음 Claude Code 대화에서 모든 컨텍스트 유지됨
echo.
echo  사용법:
echo    1. 다른 PC에서: git pull 후 이 bat 실행
echo    2. 작업 종료 시: 이 bat 실행 후 git push
echo.
pause
