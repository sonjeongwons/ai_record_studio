---
name: memory-maintenance-rule
description: 코드/결정 변경 시 CLAUDE.md + .claude-memory/*.md + bat 스크립트 3종 모두 현행화 필수
type: feedback
---

코드 변경이나 프로젝트 결정사항 변경 시 **3곳 모두** 업데이트할 것:

1. **CLAUDE.md** — 프로젝트 지시사항 (규칙/제약/현황 인라인)
2. **`.claude-memory/*.md`** — 상세 메모리 파일 (git에 포함, bat으로 내부 메모리에 복사)
3. **`~/.claude/projects/<hash>/memory/`** — Claude Code 내부 메모리 (bat 또는 직접 cp)

**Why:** 사용자는 git pull + bat 실행으로 어떤 PC에서든 완벽한 맥락 복원을 원함.
CLAUDE.md는 프로젝트 지시사항으로 로드되고, 내부 메모리는 자동 메모리로 대화 컨텍스트에 주입됨.
두 채널이 다르므로 둘 다 유지해야 가장 풍부한 맥락 제공.

**How to apply:**
- 코드 변경 후 커밋 시: 관련 .claude-memory/*.md 업데이트 + CLAUDE.md 현황 반영
- 새로운 기술 결정 시: .claude-memory/에 project_*.md 추가 + CLAUDE.md 이력 섹션 반영
- 사용자 피드백 시: .claude-memory/에 feedback_*.md 추가/수정
- 항상 .claude-memory/ 변경 후 내부 메모리(~/.claude/)에도 동기화 (cp 명령)
