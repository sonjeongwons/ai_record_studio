#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claude Code 메모리 복원 스크립트 (Mac/Linux)
#
# 사용법: 다른 PC에서 git pull 후 실행
#   bash setup-claude-memory.sh
#
# 같은 PC의 다른 대화창이면 CLAUDE.md가 자동으로 읽히므로
# 이 스크립트 실행이 필수는 아닙니다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/.claude-memory"
CLAUDE_BASE="$HOME/.claude/projects"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] .claude-memory/ 디렉토리를 찾을 수 없습니다."
    echo "  이 스크립트는 프로젝트 루트에서 실행해야 합니다."
    exit 1
fi

if [ ! -d "$CLAUDE_BASE" ]; then
    echo "Claude Code 프로젝트 디렉토리가 없습니다."
    echo ""
    echo "다음 단계를 따라주세요:"
    echo "  1. 이 프로젝트 폴더에서 'claude' 명령을 한번 실행"
    echo "  2. 아무 메시지나 보내고 종료"
    echo "  3. 이 스크립트 재실행"
    exit 1
fi

FOUND_DIR=""

# Strategy 1: 디렉토리 이름으로 찾기
for dir in "$CLAUDE_BASE"/*ai*record*studio* "$CLAUDE_BASE"/*ai-record* 2>/dev/null; do
    if [ -d "$dir" ]; then
        FOUND_DIR="$dir/memory"
        break
    fi
done

# Strategy 2: MEMORY.md 내용으로 프로젝트 식별
if [ -z "$FOUND_DIR" ]; then
    for dir in "$CLAUDE_BASE"/*/; do
        if [ -f "$dir/memory/MEMORY.md" ]; then
            if grep -q "ai_record_studio\|AI Voice Studio" "$dir/memory/MEMORY.md" 2>/dev/null; then
                FOUND_DIR="$dir/memory"
                break
            fi
        fi
    done
fi

# Strategy 3: memory 디렉토리가 있는 아무 프로젝트
if [ -z "$FOUND_DIR" ]; then
    for dir in "$CLAUDE_BASE"/*/; do
        if [ -d "$dir/memory" ]; then
            FOUND_DIR="$dir/memory"
            break
        fi
    done
fi

if [ -z "$FOUND_DIR" ]; then
    echo "기존 Claude Code 메모리 디렉토리를 찾지 못했습니다."
    echo "프로젝트 폴더에서 'claude' 명령을 한번 실행한 후 재시도해주세요."
    exit 1
fi

echo ""
echo "[INFO] 메모리 복원 대상: $FOUND_DIR"
echo ""

mkdir -p "$FOUND_DIR"

COUNT=0
for f in "$SOURCE_DIR"/*.md; do
    if [ -f "$f" ]; then
        cp "$f" "$FOUND_DIR/"
        echo "  + $(basename "$f")"
        COUNT=$((COUNT + 1))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[OK] Claude Code 메모리 복원 완료! (${COUNT}개 파일)"
echo ""
echo "이제 'claude' 명령을 실행하면 이전 대화 맥락이 유지됩니다."
echo "CLAUDE.md + HANDOFF.md는 자동으로 읽히며,"
echo ".claude-memory/ 의 상세 메모리도 로드됩니다."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
