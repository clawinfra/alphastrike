# AlphaStrike Autonomous Agent Instructions

You are an autonomous coding agent building the AlphaStrike trading bot. Follow these instructions precisely.

## On Startup

1. **Check for handoff.json** - If exists, resume from handoff instruction
2. **Read progress.txt** - Pay attention to "Codebase Patterns" section
3. **Read prd.json** - Find highest priority story where `passes: false`
4. **Read docs/ARCHITECTURE.md** - Reference for implementation details

## Implementation Protocol

### Before Coding
1. Read the story's acceptance criteria carefully
2. Check related files in the codebase for patterns
3. Reference docs/ARCHITECTURE.md for specifications

### While Coding
1. Work on ONE story at a time
2. Follow Python conventions: type hints, dataclasses, async
3. Use structured logging (JSON format)
4. Run typecheck frequently: `uv run pyright src/`
5. Commit logical units of work

### After Completing a Story
1. Run quality checks:
   ```bash
   uv run pyright src/
   uv run pytest tests/ -v
   ```
2. If checks pass, commit: `git commit -m "feat: [Story ID] - [Story Title]"`
3. Update prd.json: set story's `passes: true`
4. Append to progress.txt with learnings
5. Check if ALL stories complete -> output `<promise>COMPLETE</promise>`
6. Otherwise, continue to next story

## Context Threshold Detection

Watch for these signs you're approaching context limits:
- Difficulty recalling earlier conversation details
- Responses becoming shorter or less detailed
- Needing to re-read files you recently viewed

When detected:
1. Stop at the nearest safe checkpoint
2. Commit any complete work
3. Write handoff.json with current state
4. Output `<handoff>CONTEXT_THRESHOLD</handoff>`

## Handoff JSON Format

```json
{
  "timestamp": "2026-01-30T10:30:00Z",
  "reason": "context_threshold",
  "current_story": {
    "id": "US-XXX",
    "title": "Story title",
    "progress_percent": 65,
    "status": "implementing"
  },
  "work_in_progress": {
    "files_modified": ["src/file1.py", "src/file2.py"],
    "uncommitted_changes": "Description of uncommitted work",
    "last_completed_step": "What was just finished",
    "next_steps": ["Step 1", "Step 2", "Step 3"]
  },
  "context_learned": ["Pattern 1", "Pattern 2"],
  "blockers": [],
  "handoff_instruction": "Continue implementing US-XXX - description of where to pick up"
}
```

## Progress Entry Format

Append to progress.txt after each story:

```
## [Date] - [Story ID]: [Story Title]
- What was implemented
- Files changed
- **Learnings:**
  - Pattern discovered
  - Gotcha encountered
---
```

## Quality Standards

- All functions have type hints
- Async functions use `async def`
- Use `logging.getLogger(__name__)` per module
- Dataclasses for structured data
- pyright/mypy typecheck must pass
- Unit tests for core logic

## Key Files to Reference

- `docs/ARCHITECTURE.md` - Full system architecture
- `docs/PRD.md` - Product requirements
- `prd.json` - Story tracking
- `progress.txt` - Learnings and patterns
- `src/core/config.py` - Central configuration (after US-001)
