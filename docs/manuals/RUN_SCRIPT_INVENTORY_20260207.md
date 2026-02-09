# Run Script Inventory (2026-02-07)

## Scope

- Executable shell scripts (`*.sh`) in project root and `scripts/`
- Python entry scripts in `scripts/` (`if __name__ == "__main__"`)

## Summary

- Canonical full pipeline entry is `python main.py --full` (wrapper script removed).
- Contract verification script `scripts/check_execution_contract.sh` is active and referenced by ADR/work orders/handoff docs.
- Legacy utility scripts removed (2026-02-10):
  - `scripts/check_gold_data.py` (ad-hoc ticker probe, no active references)
  - `scripts/visualize_agents.py` (standalone dashboard generator, no active references)
- Frontend one-time merge script was removed:
  - `scripts/merge_frontend.sh` (deleted on 2026-02-07)
  - Reason: source tree `frontend_steps/` no longer exists and script had destructive `rm -rf frontend`.

## Executable Shell Scripts

| Script | Reference Count* | Status | Action |
|---|---:|---|---|
| `scripts/check_execution_contract.sh` | 17 | Active | Keep (contract gate) |
| `run_all.sh` | 1 | Active | Keep (local dashboard bring-up) |
| `stop_all.sh` | 1 | Active | Keep (local dashboard stop) |
| `scripts/delegate_general_lane.sh` | 2 | Active | Keep (work-order lane wrapper) |
| `scripts/setup_scheduler.sh` | 0 | Manual utility | Keep (optional cron setup) |
| `scripts/merge_frontend.sh` | 1 | Obsolete | Removed |

## Python Entry Scripts (`scripts/*.py`)

| Script | Reference Count* | Status | Action |
|---|---:|---|---|
| `scripts/daily_collector.py` | 10 | Active | Keep |
| `scripts/run_backtest.py` | 9 | Active | Keep |
| `scripts/scheduler.py` | 7 | Active | Keep |
| `scripts/daily_analysis.py` | 5 | Active | Keep |
| `scripts/prepare_historical_data.py` | 2 | Active | Keep |
| `scripts/delegate_general_lane.py` | 1 | Active | Keep |
| `scripts/validate_integration_design.py` | 1 | Manual utility | Keep |
| `scripts/validate_methodology.py` | 1 | Manual utility | Keep |
| `scripts/convert_md_to_html.py` | 1 | Manual utility | Keep (used by `generate_final_report.py`) |
| `scripts/generate_final_report.py` | 6 | Manual utility | Keep (reporting utility) |
| `scripts/check_gold_data.py` | 0 | Obsolete | Removed |
| `scripts/visualize_agents.py` | 0 | Obsolete | Removed |

\* Reference count was measured with repository text search excluding archive-heavy paths (`archive/**`, `docs/archive/**`) and excluding self-file matches.

## Next Cleanup Candidates

1. Decide whether `scripts/setup_scheduler.sh` should move to `docs/manuals/` example-only snippet.
2. Review low-reference (`<=1`) manual utilities for runbook promotion vs removal.
