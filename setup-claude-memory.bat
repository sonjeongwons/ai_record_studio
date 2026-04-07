@echo off
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Claude Code 메모리 + 설정 복원 스크립트 (Windows)
REM
REM 사용법: 다른 PC에서 git pull 후 이 스크립트를 더블클릭
REM  1. .claude-sync/memory/ → Claude Code 내부 메모리에 복사
REM  2. .claude/settings.json → 프로젝트 설정 자동 적용
REM  3. 다음 Claude 대화에서 이전 맥락이 자동 로드됨
REM
REM 대화 자체는 PC별 로컬이지만, CLAUDE.md + 메모리 파일로
REM 프로젝트 맥락, 기술 결정, 사용자 선호도가 모두 복원됩니다.
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SYNC_MEMORY=%SCRIPT_DIR%.claude-sync\memory"
set "CLAUDE_BASE=%USERPROFILE%\.claude\projects"

echo.
echo ============================================
echo  AI Voice Studio - Claude Memory Restore
echo ============================================
echo.

REM ── 1. 소스 확인 ──
if not exist "%SYNC_MEMORY%" (
    echo [ERROR] .claude-sync\memory\ 를 찾을 수 없습니다.
    echo   git pull이 완료되었는지 확인하세요.
    goto :error
)

echo [1/3] 메모리 소스: %SYNC_MEMORY%
for %%F in ("%SYNC_MEMORY%\*.md") do (
    echo       - %%~nxF
)
echo.

REM ── 2. Claude Code 프로젝트 디렉토리 찾기 ──
set "TARGET_DIR="

REM Strategy 1: ai_record_studio가 포함된 디렉토리
for /d %%D in ("%CLAUDE_BASE%\*ai*record*studio*") do (
    set "TARGET_DIR=%%D\memory"
    goto :found
)

REM Strategy 2: c--ai-record-studio (Claude Code가 자동 생성하는 패턴)
for /d %%D in ("%CLAUDE_BASE%\c--ai-record*") do (
    set "TARGET_DIR=%%D\memory"
    goto :found
)

REM Strategy 3: MEMORY.md 내용으로 식별
for /d %%D in ("%CLAUDE_BASE%\*") do (
    if exist "%%D\memory\MEMORY.md" (
        findstr /c:"ai_record_studio" /c:"AI Voice Studio" "%%D\memory\MEMORY.md" >nul 2>&1
        if not errorlevel 1 (
            set "TARGET_DIR=%%D\memory"
            goto :found
        )
    )
)

REM Strategy 4: 새로 생성
echo [INFO] 기존 메모리 디렉토리를 찾지 못했습니다.
echo        새 디렉토리를 생성합니다...
set "TARGET_DIR=%CLAUDE_BASE%\c--ai-record-studio\memory"
mkdir "%TARGET_DIR%" 2>nul
goto :found

:found
echo [2/3] 복원 대상: %TARGET_DIR%
echo.

REM 디렉토리 생성 (없으면)
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

REM ── 3. 메모리 파일 복사 ──
set "COUNT=0"
for %%F in ("%SYNC_MEMORY%\*.md") do (
    copy /Y "%%F" "%TARGET_DIR%\" >nul 2>&1
    set /a COUNT+=1
    echo       [OK] %%~nxF
)

echo.
echo ============================================
echo  [3/3] 복원 완료! (%COUNT%개 메모리 파일)
echo ============================================
echo.
echo  자동 로드되는 항목:
echo   - CLAUDE.md       : 전체 프로젝트 맥락
echo   - HANDOFF.md      : 아키텍처 결정사항
echo   - MEMORY.md       : 메모리 인덱스 (자동 로드)
echo   - 메모리 파일 %COUNT%개 : 상세 기억 (필요시 참조)
echo   - settings.json   : 프로젝트 설정 + 권한
echo.
echo  이제 이 폴더에서 claude 명령을 실행하면
echo  이전 대화 맥락이 모두 유지됩니다.
echo.
echo  참고: 대화 히스토리는 PC별 로컬 저장이지만,
echo  CLAUDE.md + 메모리로 프로젝트 맥락이 완전히 복원됩니다.
echo ============================================
echo.
pause
exit /b 0

:error
echo.
echo 복원 실패. 위의 에러 메시지를 확인하세요.
pause
exit /b 1
