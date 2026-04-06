@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   Claude Code Memory Sync - PULL
echo   GitHub → 현재 PC (메모리 + 설정 복원)
echo ============================================
echo.

:: ─── 프로젝트 경로 설정 ───
set "REPO_DIR=%~dp0"
set "SYNC_DIR=%REPO_DIR%.claude-sync"
set "MEMORY_SYNC=%SYNC_DIR%\memory"
set "CLAUDE_HOME=%USERPROFILE%\.claude"

:: ─── Git Pull ───
echo [1/5] 최신 코드 가져오는 중...
cd /d "%REPO_DIR%"
git pull
if !errorlevel! neq 0 (
    echo [ERROR] Git pull 실패
    pause
    exit /b 1
)
echo       Pull 완료!
echo.

:: ─── 동기화 파일 확인 ───
if not exist "%MEMORY_SYNC%\MEMORY.md" (
    echo [ERROR] 동기화 데이터가 없습니다.
    echo         먼저 다른 PC에서 sync-push.bat 를 실행하세요.
    pause
    exit /b 1
)

:: ─── 프로젝트 키 결정 ───
:: 현재 repo 경로 기반으로 Claude 프로젝트 키 생성
:: Claude Code는 경로를 c--path-to-project 형식으로 변환
set "REPO_PATH=%REPO_DIR:~0,-1%"
:: 드라이브 문자 소문자 변환 + 경로 구분자 변환
set "PROJECT_KEY="

:: 기존 프로젝트 키 자동 감지 (이미 Claude를 사용한 적 있는 경우)
for /d %%D in ("%CLAUDE_HOME%\projects\*") do (
    if exist "%%D\memory" (
        set "PROJECT_KEY=%%~nxD"
        set "MEMORY_DST=%%D\memory"
    )
)

:: 프로젝트 키를 못 찾으면 새로 생성
if "%PROJECT_KEY%"=="" (
    echo [INFO] 기존 Claude 프로젝트를 찾을 수 없습니다.
    echo        Claude Code를 먼저 한 번 실행하여 프로젝트를 초기화합니다...
    echo.

    :: Claude Code를 짧게 실행하여 프로젝트 디렉토리 자동 생성
    echo        아래 명령을 실행한 후 바로 종료 (Ctrl+C) 하세요:
    echo        cd /d "%REPO_DIR%" ^&^& claude
    echo.
    echo        종료 후 이 스크립트를 다시 실행하세요.
    pause
    exit /b 0
)

echo [2/5] 프로젝트 키: %PROJECT_KEY%
echo       메모리 대상: %MEMORY_DST%
echo.

:: ─── 메모리 디렉토리 생성 ───
if not exist "%MEMORY_DST%" mkdir "%MEMORY_DST%"

:: ─── 메모리 파일 복원 ───
echo [3/5] 메모리 파일 복원 중...
xcopy "%MEMORY_SYNC%\*.md" "%MEMORY_DST%\" /Y /Q >nul 2>&1
echo       메모리 파일 복원 완료!

:: 복원된 파일 목록 표시
echo       복원된 파일:
for %%F in ("%MEMORY_DST%\*.md") do (
    echo         - %%~nxF
)
echo.

:: ─── 설정 파일 안내 ───
echo [4/5] 설정 참고:
echo       settings.json은 인증 토큰 포함으로 동기화 제외.
echo       Claude Code 설정은 각 PC에서 수동 구성 필요.
echo.

:: ─── 세션 정보 표시 ───
echo [5/5] 이전 세션 정보:
if exist "%SYNC_DIR%\last_session.txt" (
    type "%SYNC_DIR%\last_session.txt"
) else (
    echo       세션 정보 없음
)
echo.

echo ============================================
echo   복원 완료! 이제 Claude Code를 시작하세요.
echo ============================================
echo.
echo   사용법:
echo     cd /d "%REPO_DIR%"
echo     claude
echo.
echo   Claude가 시작되면 메모리가 자동 로드됩니다.
echo   이전 대화 컨텍스트를 불러오려면:
echo     "이전 세션에서 이어서 작업하자. SESSION_CONTEXT.md 읽어줘"
echo   라고 입력하세요.
echo.
pause
