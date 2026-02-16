# Context Management Guide for EIMAS Agent Teams

Last Updated: 2026-02-16

## Purpose

This document defines when agent teams should clear context, compress conversations, and manage token usage efficiently during collaborative work on the EIMAS project.

---

## 1. Context Window Behavior

### Auto-Compression

Claude Code automatically compresses prior messages as conversations approach context limits. This means:
- ✅ **No manual intervention needed** for long conversations within the same task
- ✅ **Conversation is not limited** by context window
- ⚠️ **Compression has costs** (both tokens and potential information loss)

### When Auto-Compression is Sufficient

**✅ Keep working without `/clear`:**
- Continuing work on the same feature/bug
- Sequential file edits in the same module
- Debugging within the same subsystem
- Normal back-and-forth clarifications
- Team coordination on a single task

**Example (GOOD - no clear needed):**
```
Alpha Lead: "Implement portfolio allocation engine"
Codex Executor: [creates lib/allocation_engine.py]
Alpha Lead: "Now add Black-Litterman support"
Codex Executor: [edits same file]
Beta Codex: [reviews changes]
→ NO CLEAR NEEDED (same feature, related context)
```

---

## 2. Critical `/clear` Signals

### Signal 1: Project/Subsystem Switch ⚠️ HIGH PRIORITY

**When:**
- Switching from EIMAS → other project
- Moving from pipeline code → frontend code
- Changing from data collection → AI agents
- Different CLAUDE.md context

**Why:**
- Wrong project instructions persist
- Confuses file paths and dependencies
- Mixes incompatible architectural patterns

**Example (REQUIRES CLEAR):**
```bash
# Alpha Lead working on backend
[task: optimize Phase 2 caching]
... 50 messages ...

# Now switching to frontend
/clear
[task: add chart to Next.js dashboard]
```

### Signal 2: Context Pollution/Confusion ⚠️ MEDIUM PRIORITY

**When:**
- Same task attempted 3+ different ways
- Conflicting approaches discussed
- Agent mentions wrong file paths
- Response references deprecated code
- Team members getting "stuck" in loops

**Why:**
- Accumulated failed attempts cloud judgment
- Agent assumes context from previous failures
- Increases hallucination risk

**Example (REQUIRES CLEAR):**
```
Agent: "Try fixing test by mocking API" [fails]
Agent: "Try patching collector instead" [fails]
Agent: "Try refactoring entire module" [fails]
→ /clear + "Fresh approach: what is the root cause?"
```

### Signal 3: Long Debugging Sessions ⚠️ LOW PRIORITY

**When:**
- 30+ messages on same bug
- Error traces and logs piling up
- Performance degradation noticed
- Agent responses getting slower

**Why:**
- Compressed context loses nuance
- Error traces dominate token budget
- Fresh perspective helps

**Example (CONSIDER CLEAR):**
```bash
# After 35 messages debugging import error
/clear
[paste: final error message + relevant code only]
"This import fails. Files are X, Y, Z. What's wrong?"
```

### Signal 4: Sensitive Data Cleanup 🔒 CRITICAL

**When:**
- API keys accidentally pasted
- Credentials shown in error logs
- Internal paths/secrets exposed
- Before switching to public context

**Why:**
- Security requirement
- Prevent leakage to other agents/sessions

**Example (IMMEDIATE CLEAR):**
```bash
Agent: "What's this error?" [pastes logs with ANTHROPIC_API_KEY=sk-ant-...]
→ /clear (IMMEDIATELY)
→ Rotate API key
```

### Signal 5: Daily Session Reset ⏰ OPTIONAL

**When:**
- Starting new work day
- Previous session ended 8+ hours ago
- Different team members active

**Why:**
- Clean slate for new priorities
- Avoid context drift across time
- Better for async collaboration

**Example (RECOMMENDED):**
```bash
# Monday morning, continuing Friday's work
/clear
[read: .claude/sync/tasks/task_123.json]
"Continuing task 123: implement allocation engine"
```

---

## 3. Signal Detection Checklist

Before each agent turn, quickly check:

```
❓ New project/subsystem?        → YES → /clear
❓ 3+ failed approaches?          → YES → /clear
❓ API key in conversation?       → YES → /clear (URGENT)
❓ 30+ messages on same issue?    → MAYBE → /clear
❓ Agent confused about files?    → YES → /clear
❓ New work day started?          → OPTIONAL → /clear
```

---

## 4. Team-Specific Guidance

### Alpha Team (Development)

**Alpha Lead:**
- ✅ Clear when switching between tracks (B/C/F)
- ✅ Clear when starting new feature
- ❌ Don't clear during multi-file refactoring

**Codex Executor:**
- ✅ Clear before large refactors (to avoid context bias)
- ❌ Don't clear during feature iterations

**Alpha Codex:**
- ✅ Clear if review context > 5000 lines
- ✅ Use `codex review` output as fresh context

### Beta Team (QA/Security)

**Beta Lead:**
- ⚠️ ALWAYS start with fresh context (queue-based)
- ✅ Clear after each task batch

**Beta Codex:**
- ✅ Clear before each review batch
- ✅ Use minimal report template (avoid context bloat)

**Beta QA:**
- ✅ Clear after each validation run
- ❌ Don't accumulate test logs

---

## 5. Token Budget Optimization

### Strategy: Minimize Claude, Maximize Codex

```
Traditional (Claude does everything):
Planning:     500 tokens
Reading:    3,000 tokens (7 files × 430 lines avg)
Analysis:   2,000 tokens
Coding:     4,000 tokens
Review:     1,000 tokens
-------------------------------
Total:     10,500 tokens/task

Optimized (Codex-first):
Planning:     500 tokens (Claude Opus)
Delegation:    50 tokens (Haiku → Codex)
Execution:     0 tokens (Codex CLI, not Claude!)
Review:       100 tokens (Haiku → Codex review)
-------------------------------
Total:        650 Claude tokens/task

Savings:     93.8%
```

### When to Use Each Tool

| Task | Tool | Why |
|------|------|-----|
| Read code | Codex CLI | Doesn't use Claude tokens |
| Edit code | Codex exec | Doesn't use Claude tokens |
| Review code | Codex review | Doesn't use Claude tokens |
| Strategy | Claude Opus | Need reasoning |
| Planning | Claude Opus | Need creativity |
| Coordination | Claude Haiku | Fast + cheap |

### Context Compression Warning Signs

**⚠️ Compression costs accumulating:**
- Conversation > 100 messages
- Agent mentions "as we discussed earlier" (info loss)
- Repeated clarifications needed
- Response latency increasing

**Action:** `/clear` + summarize key decisions in task JSON

---

## 6. Best Practices

### DO ✅

1. **Clear proactively** when switching contexts
2. **Use task JSONs** to preserve state across clears
3. **Paste minimal context** after clearing (not full history)
4. **Delegate to Codex** for code reading/editing
5. **Keep Beta reports minimal** (template only)

### DON'T ❌

1. **Don't clear mid-feature** (loses progress)
2. **Don't paste full conversation** after clear (defeats purpose)
3. **Don't read code with Claude** (use Codex CLI)
4. **Don't accumulate test logs** (run → report → clear)
5. **Don't clear every 10 messages** (auto-compression handles it)

---

## 7. Agent Team Coordination

### Pre-Clear Checklist (Team Lead)

Before issuing `/clear` in team context:

```bash
# 1. Save state to task JSON
cat > .claude/sync/tasks/task_123.json <<EOF
{
  "id": "task_123",
  "state": "in_progress",
  "files_changed": ["lib/allocation_engine.py"],
  "next_steps": ["add tests", "update docs"],
  "blocked_by": []
}
EOF

# 2. Commit partial work (if safe)
git add lib/allocation_engine.py
git commit -m "WIP: allocation engine baseline"

# 3. Clear
/clear

# 4. Resume with minimal context
[read: .claude/sync/tasks/task_123.json]
"Continue task 123: add tests for allocation engine"
```

### Cross-Agent Context Sharing

**Problem:** Agent A clears, Agent B loses context

**Solution:** Use shared task files

```yaml
# .claude/sync/tasks/task_123.json
id: task_123
assignee: alpha-codex
context_summary: |
  Implemented MVO + Risk Parity in lib/allocation_engine.py.
  Next: Add Black-Litterman (needs views matrix).
  Blocker: None
files:
  - lib/allocation_engine.py (new, 450 lines)
  - tests/test_allocation.py (stub)
```

---

## 8. Monitoring Context Health

### Metrics to Track

```python
# In agent logs or reports
{
  "context_health": {
    "message_count": 45,
    "estimated_tokens": 12000,
    "compression_events": 2,
    "context_switches": 0,
    "clear_recommended": false
  }
}
```

### When to Alert Team Lead

```
❗ Alert if:
- message_count > 80
- compression_events > 3
- Agent mentions "I don't recall" or "as discussed"
- Response time > 30s consistently
```

---

## 9. Examples

### Example 1: Good Context Hygiene

```bash
# Day 1: Start feature
Alpha Lead: "Implement GC-HRP portfolio"
Codex Executor: [creates lib/graph_clustered_portfolio.py]
Beta Codex: [reviews, passes]

# Day 2: Continue (no clear needed)
Alpha Lead: "Add MST clustering"
Codex Executor: [edits same file]

# Day 3: New feature (clear recommended)
/clear
Alpha Lead: "New task: IBKR broker integration"
```

### Example 2: Pollution Recovery

```bash
# After multiple failed attempts
Agent: "ImportError persists after 5 fixes"
[context full of old error traces]

# Recovery
/clear
[paste: current error only]
"Fresh look: this import fails. Current file structure is X."
```

### Example 3: Team Handoff

```bash
# Alpha completes, hands to Beta
Alpha: [commits code, updates task JSON]
Beta Lead: /clear  # Start fresh
Beta Lead: [reads task JSON]
Beta Codex: [runs codex review --uncommitted]
```

---

## 10. Quick Reference

```
┌─────────────────────────────────────────────────────┐
│ CLEAR DECISION TREE                                 │
├─────────────────────────────────────────────────────┤
│ Switching projects?           → CLEAR NOW           │
│ API key exposed?              → CLEAR IMMEDIATELY   │
│ 3+ failed approaches?         → CLEAR + RETHINK     │
│ 30+ messages same bug?        → CONSIDER CLEAR      │
│ New work day?                 → OPTIONAL CLEAR      │
│ Mid-feature progress?         → DON'T CLEAR         │
│ Normal back-and-forth?        → DON'T CLEAR         │
└─────────────────────────────────────────────────────┘
```

---

## Related Documents

- **Token optimization:** `.claude/OPERATING_MODEL.md`
- **Codex delegation:** `.claude/CODEX_STRATEGY.md`
- **Task management:** `.claude/sync/tasks/README.md`
- **Agent configs:** `.claude/agents/*.yaml`

---

*For questions or updates, see: `.claude/OPERATING_MODEL.md`*
