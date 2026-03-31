@echo off
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Claude Code 메모리 복원 스크립트 (Windows)
REM 다른 PC에서 git clone 후 이 스크립트를 실행하면
REM Claude Code가 이전 대화 맥락을 이어받을 수 있습니다.
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

setlocal enabledelayedexpansion

set "SOURCE_DIR=%~dp0.claude-memory"
set "CLAUDE_BASE=%USERPROFILE%\.claude\projects"

if not exist "%SOURCE_DIR%" (
    echo Error: .claude-memory\ 디렉토리를 찾을 수 없습니다.
    exit /b 1
)

if not exist "%CLAUDE_BASE%" (
    echo Claude Code 프로젝트 디렉토리가 없습니다.
    echo 이 프로젝트에서 claude 명령을 먼저 한번 실행해주세요.
    exit /b 1
)

REM Claude 메모리 디렉토리 찾기
set "FOUND_DIR="
for /d %%D in ("%CLAUDE_BASE%\*") do (
    if exist "%%D\memory\MEMORY.md" (
        findstr /c:"ai_record_studio" /c:"AI Voice Studio" /c:"voice-studio" "%%D\memory\MEMORY.md" >nul 2>&1
        if not errorlevel 1 (
            set "FOUND_DIR=%%D\memory"
            goto :found
        )
    )
)

REM 못 찾은 경우 memory 디렉토리가 있는 아무 프로젝트 사용
for /d %%D in ("%CLAUDE_BASE%\*") do (
    if exist "%%D\memory" (
        set "FOUND_DIR=%%D\memory"
        goto :found
    )
)

echo Claude Code 메모리 디렉토리를 찾지 못했습니다.
echo.
echo 다음 단계를 따라주세요:
echo   1. 이 프로젝트에서 'claude' 명령을 한번 실행
echo   2. 종료 후 이 스크립트 재실행
exit /b 1

:found
echo 메모리 복원 대상: %FOUND_DIR%
copy /Y "%SOURCE_DIR%\*.md" "%FOUND_DIR%\" >nul
echo.
echo [OK] Claude Code 메모리가 복원되었습니다!
echo   이제 claude 명령을 실행하면 이전 대화 맥락이 유지됩니다.
pause
