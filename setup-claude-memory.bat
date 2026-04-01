@echo off
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Claude Code 메모리 복원 스크립트 (Windows)
REM
REM 사용법: 다른 PC에서 git pull 후 이 스크립트를 실행
REM  1. .claude-memory/ 의 메모리 파일을 Claude Code 내부에 복사
REM  2. 다음 Claude Code 대화에서 이전 맥락이 자동 로드됨
REM
REM 같은 PC의 다른 대화창이면 CLAUDE.md가 자동으로 읽히므로
REM 이 스크립트 실행이 필수는 아닙니다 (하지만 실행하면 더 풍부한 맥락 제공).
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

setlocal enabledelayedexpansion

set "SOURCE_DIR=%~dp0.claude-memory"
set "CLAUDE_BASE=%USERPROFILE%\.claude\projects"

if not exist "%SOURCE_DIR%" (
    echo [ERROR] .claude-memory\ 디렉토리를 찾을 수 없습니다.
    echo   이 스크립트는 프로젝트 루트에서 실행해야 합니다.
    exit /b 1
)

if not exist "%CLAUDE_BASE%" (
    echo Claude Code 프로젝트 디렉토리가 없습니다.
    echo.
    echo 다음 단계를 따라주세요:
    echo   1. 이 프로젝트 폴더에서 'claude' 명령을 한번 실행
    echo   2. 아무 메시지나 보내고 종료
    echo   3. 이 스크립트 재실행
    exit /b 1
)

REM ── Strategy 1: 프로젝트 이름이 포함된 디렉토리 찾기
set "FOUND_DIR="
for /d %%D in ("%CLAUDE_BASE%\*ai*record*studio*") do (
    set "FOUND_DIR=%%D\memory"
    goto :found
)

REM ── Strategy 2: MEMORY.md 내용으로 프로젝트 식별
for /d %%D in ("%CLAUDE_BASE%\*") do (
    if exist "%%D\memory\MEMORY.md" (
        findstr /c:"ai_record_studio" /c:"AI Voice Studio" /c:"voice-studio" "%%D\memory\MEMORY.md" >nul 2>&1
        if not errorlevel 1 (
            set "FOUND_DIR=%%D\memory"
            goto :found
        )
    )
)

REM ── Strategy 3: 가장 최근 수정된 프로젝트의 memory 디렉토리
set "NEWEST="
set "NEWEST_TIME=0"
for /d %%D in ("%CLAUDE_BASE%\*") do (
    if exist "%%D\memory" (
        set "FOUND_DIR=%%D\memory"
        goto :found
    )
)

REM ── Strategy 4: 새 memory 디렉토리 생성
echo 기존 Claude Code 메모리 디렉토리를 찾지 못했습니다.
echo 프로젝트 폴더에서 'claude' 명령을 한번 실행한 후 재시도해주세요.
exit /b 1

:found
echo.
echo [INFO] 메모리 복원 대상: %FOUND_DIR%
echo.

REM memory 디렉토리가 없으면 생성
if not exist "%FOUND_DIR%" mkdir "%FOUND_DIR%"

REM 모든 .md 파일 복사
set "COUNT=0"
for %%F in ("%SOURCE_DIR%\*.md") do (
    copy /Y "%%F" "%FOUND_DIR%\" >nul
    set /a COUNT+=1
    echo   + %%~nxF
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo [OK] Claude Code 메모리 복원 완료! (%COUNT%개 파일)
echo.
echo 이제 'claude' 명령을 실행하면 이전 대화 맥락이 유지됩니다.
echo CLAUDE.md + HANDOFF.md는 자동으로 읽히며,
echo .claude-memory/ 의 상세 메모리도 로드됩니다.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pause
