@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   Claude Code Memory Sync - PUSH
echo   현재 PC → GitHub → 다른 PC
echo ============================================
echo.

:: ─── 프로젝트 경로 설정 ───
set "REPO_DIR=%~dp0"
set "SYNC_DIR=%REPO_DIR%.claude-sync"
set "MEMORY_SYNC=%SYNC_DIR%\memory"
set "CLAUDE_HOME=%USERPROFILE%\.claude"

:: ─── 프로젝트 키 자동 감지 ───
:: Claude Code는 작업 디렉토리 경로를 키로 사용 (예: c--ai-record-studio)
set "PROJECT_KEY="
for /d %%D in ("%CLAUDE_HOME%\projects\*") do (
    if exist "%%D\memory\MEMORY.md" (
        :: 이 프로젝트의 메모리인지 확인 (AI Voice Studio 키워드)
        findstr /c:"AI Voice" "%%D\memory\MEMORY.md" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PROJECT_KEY=%%~nxD"
            set "MEMORY_SRC=%%D\memory"
        )
    )
)

if "%PROJECT_KEY%"=="" (
    echo [ERROR] Claude Code 메모리를 찾을 수 없습니다.
    echo         Claude Code로 이 프로젝트를 한 번 이상 사용한 후 실행하세요.
    pause
    exit /b 1
)

echo [INFO] 감지된 프로젝트 키: %PROJECT_KEY%
echo [INFO] 메모리 소스: %MEMORY_SRC%
echo.

:: ─── 동기화 디렉토리 생성 ───
if not exist "%MEMORY_SYNC%" mkdir "%MEMORY_SYNC%"

:: ─── 메모리 파일 복사 ───
echo [1/4] 메모리 파일 복사 중...
xcopy "%MEMORY_SRC%\*.md" "%MEMORY_SYNC%\" /Y /Q >nul 2>&1
echo       복사 완료: %MEMORY_SYNC%

:: ─── 설정 파일 안내 ───
echo [2/4] 설정 파일 참고:
echo       settings.json은 인증 토큰이 포함되어 git 동기화 제외.
echo       다른 PC에서 Claude Code 설정은 수동으로 구성하세요.

:: ─── 현재 대화 세션 ID 기록 ───
echo [3/4] 세션 정보 기록 중...
set "SESSION_INFO=%SYNC_DIR%\last_session.txt"
echo sync_date=%date% %time% > "%SESSION_INFO%"
echo source_pc=%COMPUTERNAME% >> "%SESSION_INFO%"
echo project_key=%PROJECT_KEY% >> "%SESSION_INFO%"
echo username=%USERNAME% >> "%SESSION_INFO%"

:: 최신 JSONL 파일명 기록 (대화 트랜스크립트)
set "LATEST_JSONL="
for %%F in ("%CLAUDE_HOME%\projects\%PROJECT_KEY%\*.jsonl") do (
    set "LATEST_JSONL=%%~nxF"
)
if not "%LATEST_JSONL%"=="" (
    echo latest_transcript=%LATEST_JSONL% >> "%SESSION_INFO%"
    echo       최신 세션: %LATEST_JSONL%
)

:: ─── Git Push ───
echo [4/4] Git 커밋 및 푸시 중...
cd /d "%REPO_DIR%"
git add .claude-sync\
git add -u .claude-sync\
git diff --cached --quiet 2>nul
if !errorlevel! equ 0 (
    echo       변경사항 없음 - 이미 최신 상태
) else (
    git commit -m "sync: Claude memory + settings (%COMPUTERNAME% %date%)"
    if !errorlevel! neq 0 (
        echo [ERROR] Git 커밋 실패
        pause
        exit /b 1
    )
    git push
    if !errorlevel! neq 0 (
        echo [ERROR] Git 푸시 실패
        pause
        exit /b 1
    )
    echo       푸시 완료!
)

echo.
echo ============================================
echo   동기화 완료! 다른 PC에서 sync-pull.bat 실행하세요
echo ============================================
pause
