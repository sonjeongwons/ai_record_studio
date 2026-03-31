#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claude Code 메모리 복원 스크립트
# 다른 PC에서 git clone 후 이 스크립트를 실행하면
# Claude Code가 이전 대화 맥락을 이어받을 수 있습니다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/.claude-memory"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: .claude-memory/ 디렉토리를 찾을 수 없습니다."
    exit 1
fi

# Claude Code 메모리 디렉토리 경로 찾기
# ~/.claude/projects/ 아래에 프로젝트 경로 해시 기반 디렉토리가 생성됨
CLAUDE_BASE="$HOME/.claude/projects"

if [ ! -d "$CLAUDE_BASE" ]; then
    echo "Claude Code 프로젝트 디렉토리가 없습니다. Claude Code를 먼저 한번 실행해주세요."
    echo "  cd $(pwd) && claude"
    exit 1
fi

# 이 프로젝트와 연결된 Claude 메모리 디렉토리 찾기
# 방법 1: 기존 메모리 디렉토리가 있는지 확인
FOUND_DIR=""
for dir in "$CLAUDE_BASE"/*/memory; do
    if [ -d "$dir" ] && [ -f "$dir/MEMORY.md" ]; then
        # MEMORY.md 내용에 "AI Voice Studio" 또는 "ai_record_studio"가 포함되어 있으면 매칭
        if grep -q "ai_record_studio\|AI Voice Studio\|voice-studio" "$dir/MEMORY.md" 2>/dev/null; then
            FOUND_DIR="$dir"
            break
        fi
    fi
done

# 방법 2: 기존 디렉토리가 없으면 프로젝트 디렉토리에서 memory 폴더 생성
if [ -z "$FOUND_DIR" ]; then
    # Claude Code가 프로젝트를 인식하면 자동으로 디렉토리를 생성함
    # 먼저 빈 memory 디렉토리를 가진 프로젝트 디렉토리를 찾기
    for dir in "$CLAUDE_BASE"/*/; do
        if [ -d "$dir" ] && [ ! -d "$dir/memory" ]; then
            continue
        fi
        if [ -d "${dir}memory" ]; then
            FOUND_DIR="${dir}memory"
            break
        fi
    done
fi

# 방법 3: 그래도 못 찾으면 사용자에게 안내
if [ -z "$FOUND_DIR" ]; then
    echo "Claude Code 메모리 디렉토리를 자동으로 찾지 못했습니다."
    echo ""
    echo "다음 단계를 따라주세요:"
    echo "  1. 이 프로젝트 디렉토리에서 'claude' 명령을 한번 실행하세요"
    echo "  2. /clear 후 종료하세요"
    echo "  3. 이 스크립트를 다시 실행하세요"
    echo ""
    echo "또는 수동으로 복사하세요:"
    echo "  cp .claude-memory/*.md ~/.claude/projects/<PROJECT_HASH>/memory/"
    exit 1
fi

# 메모리 파일 복사
echo "메모리 복원 대상: $FOUND_DIR"
cp "$SOURCE_DIR"/*.md "$FOUND_DIR/"
echo ""
echo "✓ Claude Code 메모리가 복원되었습니다!"
echo "  - MEMORY.md (인덱스)"
echo "  - feedback_harness_engineering.md"
echo "  - feedback_no_unnecessary_questions.md"
echo "  - project_v13_pipeline_fixes.md"
echo "  - project_v23_hpss_revert.md"
echo ""
echo "이제 'claude' 명령을 실행하면 이전 대화 맥락이 유지됩니다."
