# EIMAS Archive Directory

Historical code and documentation preserved for reference.

## Directory Structure

```
archive/
├── deprecated/       # Deprecated lib/ modules
├── v0_generation/    # V0 SDK web app generation scripts
│   ├── generate_fix.js
│   ├── generate_step.js
│   ├── generate_with_v0.js
│   ├── generate_with_v0.mjs
│   └── generate_with_v0.sh
├── agents/           # Archived agent implementations
├── core/             # Archived core modules
├── pipeline/         # Archived pipeline code
└── docs/             # Historical documentation (19 files)
```

## Quick Reference

| What | Where | Notes |
|------|-------|-------|
| **V0 generation scripts** | `v0_generation/` | Used to bootstrap frontend |
| **Phase reports** | `docs/PHASE_*.md` | Development history |
| **TODO/Gap analysis** | `docs/todolist.md`, `gap_analysis.md` | Mostly completed |

---

## Notes (2026-02-06 cleanup)

- `archive/legacy/` and `lib/deprecated/` were removed from the live tree.
- Canonical entrypoints:
  - pipeline: `main.py`
  - API: `api.main`

*Last updated: 2026-02-06*
