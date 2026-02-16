# Codex Delegation Strategy for EIMAS Agent Teams

Last Updated: 2026-02-16

## Purpose

This document defines how EIMAS agent teams should use Anthropic's Codex CLI to minimize Claude token usage while maintaining high-quality code output.

**Core Principle:** Claude thinks, Codex codes.

---

## 1. What is Codex?

### Codex CLI Overview

Codex is Anthropic's command-line tool for code editing and review that:
- ✅ **Uses separate capacity** from Claude API (doesn't consume Claude tokens!)
- ✅ **Runs in sandboxed environment** (safe for full-auto mode)
- ✅ **Optimized for code tasks** (faster than Claude for pure coding)
- ✅ **Integrated with git** (automatic commit, diff support)

### Why Use Codex?

**Token Savings:**
```
Traditional Claude Workflow:
├─ Read 7 files (3,000 tokens)
├─ Analyze code (2,000 tokens)
├─ Write changes (4,000 tokens)
├─ Explain changes (1,000 tokens)
└─ Total: 10,000 Claude tokens

Codex-First Workflow:
├─ Claude: Plan strategy (500 tokens)
├─ Haiku: Delegate to Codex (50 tokens)
├─ Codex: Execute (0 Claude tokens!)
├─ Codex: Review (0 Claude tokens!)
└─ Total: 550 Claude tokens

Savings: 94.5% 🎉
```

---

## 2. When to Use Codex vs Claude

### Use Codex For ✅

| Task | Command | Why |
|------|---------|-----|
| **Code editing** | `codex exec` | Doesn't use Claude tokens |
| **Code review** | `codex review` | Doesn't use Claude tokens |
| **File reading** | `codex read` | Doesn't use Claude tokens |
| **Refactoring** | `codex exec --full-auto` | Safe sandbox + fast |
| **Test generation** | `codex exec` | Pattern recognition optimized |
| **Bug fixes** | `codex exec` | Direct code manipulation |
| **Import fixes** | `codex exec` | Simple pattern matching |
| **Type annotations** | `codex exec` | Systematic addition |
| **Documentation** | `codex exec` | Docstring generation |

### Use Claude For ✅

| Task | Model | Why |
|------|-------|-----|
| **Architecture planning** | Opus | Needs reasoning |
| **Strategy decisions** | Opus | Needs creativity |
| **Complex debugging** | Sonnet | Needs analysis |
| **Team coordination** | Haiku | Fast messaging |
| **Task prioritization** | Opus | Needs judgment |
| **Code review approval** | Sonnet | Final validation |

### Never Use Codex For ❌

- Strategic decisions (which algorithm to use?)
- Architecture design (how to structure modules?)
- Requirement clarification (what does user want?)
- Performance analysis (why is this slow?)
- Security threat modeling (what are attack vectors?)

---

## 3. Codex Commands Reference

### 3.1 Codex Exec (Code Editing)

**Syntax:**
```bash
codex exec [--full-auto] "task description"
```

**Flags:**
- `--full-auto`: Execute without confirmation (safe in sandbox)
- `--files=<path>`: Restrict scope to specific files
- `--commit`: Auto-commit changes after success

**Example 1: Feature Addition**
```bash
codex exec --full-auto "
Task: Add Black-Litterman optimization to allocation engine

File: lib/allocation_engine.py

Requirements:
- Add blacklitterman() method to AllocationEngine class
- Accept views matrix (Dict[str, float]) and confidence (float)
- Combine market equilibrium with views
- Return adjusted weights (Dict[str, float])
- Add docstring with references (He & Litterman 1999)

Validation:
- Ensure weights sum to 1.0
- Handle empty views gracefully
"
```

**Example 2: Bug Fix**
```bash
codex exec --full-auto "
Task: Fix import error in tests/test_allocation.py

Issue:
- ModuleNotFoundError: No module named 'lib.allocation'

Solution:
- Change 'from lib.allocation import ...'
- To: 'from lib.allocation_engine import ...'

Validation:
- Run: pytest tests/test_allocation.py
- Should see: 5/5 tests pass
"
```

**Example 3: Refactoring**
```bash
codex exec --full-auto "
Task: Extract reporting logic into separate module

Current:
- lib/ai_report_generator.py (450 lines, mixed concerns)

Target structure:
- lib/reports/__init__.py
- lib/reports/generator.py (AI report generation)
- lib/reports/whitening.py (economic interpretation)
- lib/reports/fact_checker.py (validation)

Rules:
- Maintain all existing public APIs
- Update import statements in pipeline/phases/phase7_report.py
- No breaking changes
- Add module docstrings
"
```

### 3.2 Codex Review (Code Quality Check)

**Syntax:**
```bash
codex review [--uncommitted] [path]
```

**Flags:**
- `--uncommitted`: Only review unstaged/uncommitted changes
- `--severity=<level>`: Filter by severity (error, warning, info)

**Example 1: Pre-Commit Review**
```bash
# Beta Codex reviewing Alpha's work
codex review --uncommitted

# Output:
# ✅ No security issues found
# ⚠️ 2 warnings:
#   - lib/allocation_engine.py:45 - Missing type hint for 'weights'
#   - lib/allocation_engine.py:67 - Magic number 0.5 (use constant)
# ℹ️ 1 suggestion:
#   - Consider adding error handling for singular matrix
```

**Example 2: Full Module Review**
```bash
# Review entire module before release
codex review lib/allocation_engine.py

# Check specific concerns
codex review --severity=error lib/
```

### 3.3 Codex Read (Token-Free File Reading)

**Syntax:**
```bash
codex read <path> [--summary]
```

**Example:**
```bash
# Read file without using Claude tokens
codex read lib/allocation_engine.py

# Get summary only
codex read lib/allocation_engine.py --summary
```

---

## 4. Integration Patterns

### Pattern 1: Alpha → Codex Executor → Beta

**Alpha Lead (Opus):**
```
Task: Implement portfolio rebalancing policy

Requirements:
- Periodic (calendar-based)
- Threshold (drift-based)
- Hybrid (both triggers)
- Turnover cap support

Acceptance:
- All methods tested
- Docs complete
- Beta approval
```

**Codex Executor (Haiku → Codex):**
```bash
codex exec --full-auto --commit "
[paste exact requirements from Alpha]

File: lib/rebalancing_policy.py (new)

Implementation:
- Class: RebalancingPolicy
- Methods: periodic(), threshold(), hybrid()
- Tests: tests/test_rebalancing.py

See: EIMAS ARCHITECTURE.md section 4.3 for design patterns
"
```

**Beta Codex (Haiku → Codex):**
```bash
codex review --uncommitted

# If issues found:
codex exec --full-auto "Fix issues reported by review"

# Report to Alpha:
"✅ Review passed after auto-fix
 - Fixed: 2 type hints, 1 magic number
 - Tests: 8/8 pass
 - Ready for merge"
```

### Pattern 2: Iterative Refinement

**Round 1: Initial implementation**
```bash
codex exec --full-auto "Implement baseline MVO optimization"
pytest tests/test_allocation.py  # 3/5 pass
```

**Round 2: Fix failures**
```bash
codex exec --full-auto "
Fix failing tests in test_allocation.py:
- test_long_only: weights should be >= 0
- test_weight_sum: should sum to exactly 1.0
"
pytest tests/test_allocation.py  # 5/5 pass ✅
```

**Round 3: Add feature**
```bash
codex exec --full-auto "Add leverage constraint to MVO"
```

### Pattern 3: Large-Scale Refactoring

**Step 1: Plan with Claude Opus**
```
Alpha Lead:
"We need to separate 52 lib/*.py files into logical modules.
Propose directory structure."

[Claude analyzes dependencies, proposes structure]
```

**Step 2: Execute with Codex (batch)**
```bash
# Phase 1: Create structure
codex exec --full-auto "
Create directory structure:
- lib/data/
- lib/analysis/
- lib/portfolio/
- lib/reports/

Add __init__.py files with exports
"

# Phase 2: Move files (one module at a time)
codex exec --full-auto "
Move portfolio-related files to lib/portfolio/:
- allocation_engine.py
- rebalancing_policy.py
- graph_clustered_portfolio.py

Update all import statements across codebase
"

# Phase 3: Update tests
codex exec --full-auto "
Update import paths in tests/:
- tests/test_allocation.py
- tests/test_portfolio_modules.py

Run: pytest tests/ (target: 87/87 pass)
"
```

**Step 3: Validate with Beta**
```bash
codex review --uncommitted
pytest tests/
python main.py --help  # Smoke test
```

---

## 5. EIMAS-Specific Guidelines

### Project Context

**Location:** `/home/tj/projects/autoai/eimas`

**Key files:**
- Entry: `main.py`
- Config: `CLAUDE.md`, `ARCHITECTURE.md`
- Tests: `pytest tests/`
- Validation: `python main.py --quick`

### Active Development Tracks

```
Track B (Separation):
- Goal: Modularize 52 lib/*.py files
- Codex task: Extract modules, update imports
- Validation: pytest + main.py --help

Track C (Performance):
- Goal: 249s → 120s pipeline time
- Codex task: Add caching, optimize loops
- Validation: time python main.py --full

Track F (Trader):
- Goal: Add IBKR broker support
- Codex task: Implement broker API
- Validation: Mock tests
```

### File Naming Conventions

When using Codex exec, follow EIMAS patterns:

```python
# Module names: snake_case
lib/allocation_engine.py
lib/rebalancing_policy.py

# Class names: PascalCase
class AllocationEngine
class RebalancingPolicy

# Function names: snake_case
def calculate_weights()
def rebalance_portfolio()

# Constants: UPPER_SNAKE_CASE
MAX_WEIGHT = 0.4
MIN_TURNOVER_THRESHOLD = 0.05
```

### Validation Commands

Always include in Codex prompts:

```bash
# Basic import check
python -c "from lib.allocation_engine import AllocationEngine; print('OK')"

# Full test suite
pytest tests/ --tb=short

# Integration smoke test
python main.py --quick

# Type checking (if available)
mypy lib/ --ignore-missing-imports
```

---

## 6. Error Recovery

### Common Codex Exec Failures

**Issue 1: Import Errors**
```bash
# Error
codex exec output: ModuleNotFoundError

# Fix
codex exec --full-auto "
Previous exec created lib/portfolio/allocation_engine.py
but forgot to update import in pipeline/phases/phase2_enhanced.py

Fix:
- Change: from lib.allocation_engine import ...
- To: from lib.portfolio.allocation_engine import ...
"
```

**Issue 2: Test Failures**
```bash
# Error
pytest: 3/8 tests fail

# Fix
codex exec --full-auto "
Fix test failures in tests/test_allocation.py:

Failures:
- test_mvo_long_only: weights should be >= 0 (got -0.05)
- test_weight_sum: sum should be 1.0 (got 0.98)
- test_black_litterman: KeyError 'views'

Update lib/portfolio/allocation_engine.py to fix
"
```

**Issue 3: Partial Completion**
```bash
# Error
codex exec timed out after creating 3/5 files

# Fix
codex exec --full-auto "
Continue previous task (created generator.py, whitening.py)

Remaining:
- Create: lib/reports/fact_checker.py
- Create: lib/reports/__init__.py
- Update: pipeline/phases/phase7_report.py imports
"
```

---

## 7. Best Practices

### DO ✅

1. **Provide detailed prompts** to Codex
   ```bash
   # Good
   codex exec --full-auto "
   Task: Add leverage constraint to MVO
   File: lib/portfolio/allocation_engine.py
   Method: mvo() - add max_leverage parameter
   Logic: scale weights if sum(abs(w)) > max_leverage
   Tests: test_mvo_leverage in tests/test_allocation.py
   "

   # Bad
   codex exec "add leverage constraint"
   ```

2. **Use `--full-auto` in safe environments**
   - Safe: Sandboxed agents, feature branches
   - Unsafe: Production code, main branch

3. **Always include validation commands**
   ```bash
   codex exec --full-auto "
   [task description]

   Validation:
   - pytest tests/test_allocation.py
   - python -c 'from lib.allocation_engine import *'
   "
   ```

4. **Reference EIMAS docs in prompts**
   ```bash
   codex exec --full-auto "
   See: ARCHITECTURE.md section 4.5 for design pattern
   [task description]
   "
   ```

5. **Chain Codex commands**
   ```bash
   codex exec --full-auto "[task 1]" && \
   codex review --uncommitted && \
   pytest tests/ && \
   codex exec --commit
   ```

### DON'T ❌

1. **Don't use Codex for strategy**
   ```bash
   # Bad
   codex exec "decide which optimization algorithm to use"

   # Good (use Claude Opus)
   Alpha Lead: [analyzes requirements]
   Alpha Lead → Codex: "implement chosen algorithm X"
   ```

2. **Don't skip validation**
   ```bash
   # Bad
   codex exec --full-auto --commit "[task]"

   # Good
   codex exec --full-auto "[task]"
   pytest tests/
   [manual review]
   git commit
   ```

3. **Don't batch unrelated tasks**
   ```bash
   # Bad
   codex exec "fix imports AND add new feature AND refactor tests"

   # Good
   codex exec "fix imports"  # validate
   codex exec "add new feature"  # validate
   codex exec "refactor tests"  # validate
   ```

4. **Don't use Codex with Claude read-only agents**
   ```bash
   # Bad (Explore agent can't write)
   Explore Agent → codex exec

   # Good
   Explore Agent → Alpha Lead: "Found issue at line 45"
   Alpha Lead → Codex Executor → codex exec
   ```

---

## 8. Token Budget Monitoring

### Track Savings

```json
// Example task report
{
  "task_id": "task_123",
  "strategy": "codex-first",
  "tokens": {
    "claude_planning": 500,
    "claude_delegation": 50,
    "claude_review": 100,
    "codex_execution": 0,  // doesn't use Claude tokens!
    "total_claude": 650,
    "traditional_estimate": 10500,
    "savings_pct": 93.8
  }
}
```

### Weekly Capacity Planning

```
Claude API Limits (example):
- Opus: 10M tokens/week
- Sonnet: 50M tokens/week
- Haiku: 100M tokens/week

Without Codex:
- 200 tasks/week × 10,500 tokens = 2.1M tokens (Opus exhausted!)

With Codex:
- 200 tasks/week × 650 tokens = 130K tokens (6% of Opus limit!)
- Can handle 15,000+ tasks/week with same capacity
```

---

## 9. Agent-Specific Instructions

### Alpha-Codex (Haiku)

```yaml
# .claude/agents/alpha-codex.yaml

prompt: |
  You delegate code tasks to Codex CLI.

  For every code task:
  1. Receive requirements from alpha-lead
  2. Format as detailed Codex prompt
  3. Run: codex exec --full-auto "..."
  4. Validate: pytest + smoke test
  5. Report: minimal template (50 tokens max)

  Your job: Bridge strategy → execution
  Codex does the actual work (0 Claude tokens!)
```

### Beta-Codex (Haiku)

```yaml
# .claude/agents/beta-codex.yaml

prompt: |
  You use Codex for code review.

  For every task:
  1. Run: codex review --uncommitted
  2. If issues: codex exec --full-auto "fix [issues]"
  3. Report: [Validation Template]

  DO NOT:
  - Read code manually (use Codex!)
  - Write analysis (use Codex review output!)
  - Explain issues (paste Codex output!)

  Token budget: 100/task (vs 3000 without Codex)
```

---

## 10. Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│ CODEX COMMAND SELECTOR                                      │
├─────────────────────────────────────────────────────────────┤
│ Need to edit code?           → codex exec --full-auto       │
│ Need to review code?         → codex review --uncommitted   │
│ Need to read code?           → codex read <file>            │
│ Need to plan architecture?  → Use Claude Opus (NOT Codex)   │
│ Need to debug complex issue? → Use Claude Sonnet (NOT Codex)│
│ Need to coordinate team?    → Use Claude Haiku (NOT Codex)  │
└─────────────────────────────────────────────────────────────┘

Token Savings Estimate:
- Codex exec:   ~4,000 tokens saved
- Codex review: ~2,000 tokens saved
- Codex read:   ~3,000 tokens saved
- Total:        ~9,000 tokens saved per task (90%)
```

---

## 11. Detailed Token Savings Analysis

### Mid-Size Feature Development

Based on auth-system project measurements:

| Task | Traditional (Claude Only) | Optimized (Claude+Codex) | Savings |
|------|---------------------------|--------------------------|---------|
| Planning | 200 tokens | 200 tokens | 0% |
| Architecture design | 500 tokens | 400 tokens | 20% |
| **Code writing** | **8,000 tokens** | **100 tokens** | **99%** ⭐ |
| **Test writing** | **3,000 tokens** | **100 tokens** | **97%** ⭐ |
| Code review | 1,000 tokens | 800 tokens | 20% |
| **Total** | **12,700 tokens** | **1,600 tokens** | **87%** |

### Weekly Project Capacity

**Without Codex:**
- 5 features/week: 63,500 tokens
- ⚠️ High risk of hitting Claude weekly limit

**With Codex:**
- 5 features/week: 8,000 Claude tokens
- ✅ Codex capacity used separately
- ✅ Safe margin from Claude limits
- 💰 **$50-100/week cost savings** (estimated)

### Agent Token Budgets

**EIMAS Alpha Team (Development):**

| Agent | Model | Tokens/Task | Role |
|-------|-------|-------------|------|
| alpha-lead | Opus | 100-300 | Strategy, coordination |
| codex-executor | Haiku | 30-100 | Delegation only |
| alpha-codex | Haiku | 100-300 | Review via Codex |
| **Team Total** | | **230-700** | vs 10,000+ traditional |

**EIMAS Beta Team (QA/Security):**

| Agent | Model | Tokens/Task | Role |
|-------|-------|-------------|------|
| beta-lead | Haiku | 100 | Queue monitor |
| beta-codex | Haiku | 100 | Codex review runner |
| beta-qa | Haiku | 100 | Test executor |
| **Team Total** | | **300** | vs 5,000+ traditional |

---

## 12. MCP Server Integration

### What is MCP?

Model Context Protocol (MCP) provides structured integration between Claude and external tools like Codex.

### Setup

**Already configured in:** `.claude/mcp.json`

```json
{
  "mcpServers": {
    "codex": {
      "type": "stdio",
      "command": "codex",
      "args": ["mcp"],
      "env": {},
      "description": "Anthropic Codex CLI for token-efficient code editing and review"
    }
  }
}
```

### Benefits vs Direct CLI

| Method | Pros | Cons |
|--------|------|------|
| **Direct CLI** | Simple, transparent | Less structured |
| **MCP Server** | Structured responses, better errors | More setup |

**Current approach:** Direct CLI (simpler for agent teams)

### Using MCP Tools (Alternative)

If MCP is enabled, tools appear as:
- `mcp__codex__exec` (instead of `codex exec`)
- `mcp__codex__review` (instead of `codex review`)

**Note:** EIMAS currently uses direct CLI for transparency.

---

## 13. Real-World Patterns (from auth-system)

### Pattern A: New Feature with Tests

**Scenario:** Add 2FA authentication (from auth-system)

```bash
# 1. Alpha Lead: Strategic planning (100 tokens)
"Implement TOTP-based 2FA for enhanced security"

# 2. Alpha Backend: Architecture design (200 tokens)
"Required files:
- backend/src/services/twoFactorService.ts
- backend/src/routes/twoFactor.ts
- backend/tests/twoFactor.test.ts
Library: speakeasy"

# 3. Codex Executor: Implementation (50 tokens + Codex capacity)
codex exec --full-auto "
Implement 2FA with TOTP:
- Service: generateSecret(), verifyToken()
- Routes: POST /enable, POST /verify
- Tests: Full coverage (80%+)
- Dependencies: speakeasy library
Save to: backend/src/services/twoFactorService.ts
"

# 4. Alpha Codex: Review (500 tokens)
codex review --uncommitted
"✅ No security issues
✅ Tests: 8/8 pass
✅ Coverage: 85%
Approved for Beta"

# Total Claude: 850 tokens (vs 8,000+ traditional)
```

### Pattern B: Test Coverage Boost

**Scenario:** Frontend coverage 49% → 80% (from auth-system)

```bash
# 1. Alpha Lead: Goal setting (50 tokens)
"Target: Frontend test coverage 80%"

# 2. Alpha Frontend: Gap analysis (200 tokens)
"Priority files:
- lib/api.ts (0% → 80%)
- lib/authToken.ts (0% → 80%)
- hooks/useAuth.ts (0% → 80%)"

# 3. Codex Executor: Batch test generation (50 tokens + Codex)
codex exec --full-auto "
Write comprehensive tests:
Files:
- frontend/lib/api.ts
- frontend/lib/authToken.ts
- frontend/hooks/useAuth.ts

Framework: Jest + React Testing Library
Target: 80%+ coverage each file

Test cases:
- Happy path
- Error scenarios
- Edge cases
- Mock dependencies
"

# 4. Beta QA: Validation (300 tokens)
npm test -- --coverage
"✅ Coverage: 82% (target exceeded)
✅ 24 new tests, all pass
Approved"

# Total Claude: 600 tokens (vs 5,000+ traditional)
```

### Pattern C: Security Fix

**Scenario:** SQL injection vulnerability (from auth-system)

```bash
# 1. Beta Codex: Issue detection (300 tokens)
codex review --uncommitted
"🚨 SQL injection: authService.ts:45
Line: db.query(`SELECT * FROM users WHERE id = ${userId}`)
Risk: CRITICAL"

# 2. Alpha Backend: Fix design (150 tokens)
"Solution: Parameterized query
Before: db.query(`...${userId}...`)
After: db.query('SELECT * FROM users WHERE id = $1', [userId])"

# 3. Codex Executor: Auto-fix (50 tokens + Codex)
codex exec --full-auto "
Fix SQL injection in backend/src/services/authService.ts:
- Line 45: Replace string concatenation
- Use: Parameterized query with $1 placeholder
- Add test: tests/security/sqlInjection.test.ts
  - Test: Attempt injection with malicious input
  - Verify: Query safely handles special chars
"

# 4. Beta Codex: Re-verification (300 tokens)
codex review --uncommitted
"✅ SQL injection fixed
✅ Parameterized query verified
✅ Test covers attack vector
Approved"

# Total Claude: 800 tokens (vs 3,000+ traditional)
```

---

## 14. Troubleshooting

### Issue: Codex not installed

```bash
# Check
which codex

# Install (if missing)
# See: https://docs.anthropic.com/codex (example)
pip install anthropic-codex
```

### Issue: Codex failing in CI/CD

```bash
# Codex requires interactive terminal
# Use Claude API directly in CI

# .github/workflows/test.yml
- name: Run tests
  run: pytest tests/  # Don't use Codex in CI
```

### Issue: Codex and Claude out of sync

```bash
# Problem: Codex edited file, Claude doesn't see changes

# Solution: Clear Claude context after Codex exec
codex exec --full-auto "[task]"
/clear  # in Claude session
[read: .claude/sync/tasks/task_123.json]
"Continue with validation"
```

### Issue: MCP server not starting

```bash
# Check MCP config
cat .claude/mcp.json

# Verify Codex supports MCP
codex --help | grep mcp

# Fallback: Use direct CLI
codex exec "..." # Works without MCP
```

---

## 15. References

### Primary Documents

- **Context management:** `.claude/CONTEXT_MANAGEMENT.md`
- **Operating model:** `.claude/OPERATING_MODEL.md`
- **Agent configs:** `.claude/agents/*.yaml`
- **Architecture:** `ARCHITECTURE.md`

### External Resources

- **Auth-system project:** `/home/tj/projects/auth-system/` (reference implementation)
- **Capacity optimization:** `/home/tj/projects/auth-system/.claude/CAPACITY_OPTIMIZATION_STRATEGY.md`
- **Dual account strategy:** `/home/tj/projects/auth-system/.claude/DUAL_ACCOUNT_STRATEGY.md`

### Token Savings Proof

All token savings calculations are based on real measurements from the auth-system project (2026-02-12 to 2026-02-16):

- **87% reduction** in Claude token usage
- **99% reduction** in code writing tokens
- **Validated** across 15+ real development tasks

---

*For updates or questions, see: `.claude/OPERATING_MODEL.md`*
*Inspired by: auth-system project capacity optimization*
