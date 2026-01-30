# EIMAS Archive Directory

Historical code and documentation preserved for reference.

## Directory Structure

```
archive/
├── legacy/           # Legacy Python entry points
│   ├── main_legacy.py
│   ├── main_integrated.py
│   ├── main_orchestrator.py
│   ├── run_full_analysis.py
│   ├── debug_forecast.py
│   └── dashboard.py      # Streamlit dashboard (replaced by Next.js)
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
| **Old Python dashboards** | `legacy/dashboard.py` | Replaced by Next.js frontend |
| **Alternate orchestrators** | `legacy/main_orchestrator.py` | Use `main.py` instead |
| **V0 generation scripts** | `v0_generation/` | Used to bootstrap frontend |
| **Phase reports** | `docs/PHASE_*.md` | Development history |
| **TODO/Gap analysis** | `docs/todolist.md`, `gap_analysis.md` | Mostly completed |

---

*Last updated: 2026-01-31*
