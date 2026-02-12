# EIMAS Operating Model

Last Updated: 2026-02-12

## Decisions (Locked)
- `autofix`: **allowed** (security, tests, typing)
- `default topology`: **2 accounts (Alpha + Beta)**
- `execution default`: **Codex** for code changes
- `reasoning/review`: **Opus lead** for strategy
- `reporting`: **MINIMAL** - validation results only

## Account Topology

### Account 1: Alpha (Development)
- `alpha-lead` (Opus): planning, task creation, approval
- `codex-executor` (Haiku): **PRIMARY CODER** - delegates ALL code work to `codex exec`
- `alpha-codex` (Haiku): quick `codex review` before Beta

### Account 2: Beta (QA/Security)
- `beta-lead` (Haiku): **MINIMAL** - queue monitor only
- `beta-codex` (Haiku): runs `codex review --uncommitted` (NOT manual analysis)
- `beta-qa` (Haiku): runs test commands only

**KEY: All agents use Haiku. Codex does the actual work.**

## Beta Team Optimization (CRITICAL)

### ❌ What NOT to Do
- Long narrative reports (save 50k tokens)
- Detailed issue analysis (save 20k tokens)
- Status updates/summaries (save 10k tokens)
- Multiple messages back and forth (save 15k tokens)

### ✅ What TO Do
- Run validation commands
- Report results in minimal template
- Auto-fix if allowed
- Block and escalate if needed

## Minimal Report Template

```text
[Task] <task_id>
Files: <count>
Validation: <commands>

[Results]
✅/❌ <command>: <pass/fail>
Issues: <critical count>

[Action]
Fixed: <list>
Blocked: <list>
```

## Task Contract (Schema v2)

```json
{
  "schema_version": "2",
  "id": "task_<timestamp>_<type>",
  "type": "code_review|testing|security_audit",
  "executor": "codex|claude",
  "auto_fix_allowed": true,
  "scope_files": ["path/to/file"],
  "validation_commands": [
    "pytest tests/",
    "python main.py --help"
  ],
  "acceptance_criteria": ["all tests pass"],
  "state": "pending|in_progress|completed|blocked"
}
```

## Validation-First Workflow

1. **Alpha creates task** with validation commands
2. **Beta runs commands** (no analysis)
3. **Beta reports results** (minimal template)
4. **Beta auto-fixes** if allowed
5. **Beta escalates** if blocked

## Token Budget (Codex-First Strategy)

| Agent | Model | Tokens/Task | What They Do |
|-------|-------|-------------|--------------|
| alpha-lead | Opus | 500 | Planning only |
| codex-executor | Haiku | 50 | Runs `codex exec` |
| beta-lead | Haiku | 100 | Runs commands |
| beta-codex | Haiku | 100 | Runs `codex review` |
| **Total Claude** | | **750** | **vs 104k before** |

**Actual work done by Codex CLI (uses Codex capacity, not Claude!)**

**Savings: 99.3%**

## Example: Tasks 1-5 Review

**Old way** (104k tokens):
- Read all 7 files
- Detailed code analysis
- 3,600-line report
- Multiple message rounds

**New way** (15k tokens):
```bash
# Run validations
pytest tests/ --tb=short
python main.py --help
npm run type-check

# Report
[Task] Tasks 1-5
Files: 7
Tests: 68/87 pass ❌
Critical: .env permissions ✅ FIXED
Block: Test failures need Alpha fix
```

## Canonical Files
- This model: `.claude/OPERATING_MODEL.md`
- Agent defs: `.claude/agents/*.yaml`
- Queue: `.claude/sync/tasks/`
- Helpers: `scripts/team-helpers/`
