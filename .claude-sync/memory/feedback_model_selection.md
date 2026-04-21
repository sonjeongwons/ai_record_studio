---
name: Model Selection for Design vs Coding
description: 분석/설계는 반드시 Opus, 코딩/구현은 반드시 Sonnet — 예외 없음
type: feedback
---
분석/설계는 **반드시 Opus 모델**, 코딩/구현은 **반드시 Sonnet 모델**을 사용한다. 예외 없음.

**Why:** 사용자가 명시적으로 규칙으로 지정 — "무조건 사용하는 것으로 md로 정해주세요"

**How to apply:**
- 분석, 아키텍처 설계, 트레이드오프 검토, 계획 수립, 리서치 → **Opus** (`model: "opus"`)
- 코드 작성, 버그 수정, 파일 편집, 테스트 작성, 리팩토링 → **Sonnet** (`model: "sonnet"`)
- Agent tool 사용 시 반드시 `model` 파라미터 명시: 설계 에이전트 `model: "opus"`, 코딩 에이전트 `model: "sonnet"`
- 한 작업 안에서 설계→구현 순서로 진행할 때: Plan 에이전트(opus) 먼저 → Code 에이전트(sonnet) 실행
