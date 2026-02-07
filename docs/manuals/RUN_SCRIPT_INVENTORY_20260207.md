# Run Script Inventory (2026-02-07)

## Scope

- Executable shell scripts (`*.sh`) in project root and `scripts/`
- Python entry scripts in `scripts/` (`if __name__ == "__main__"`)

## Summary

- Canonical full pipeline entry remains `run_all_pipeline.sh` -> `python main.py --full`.
- Contract verification script `scripts/check_execution_contract.sh` is active and referenced by ADR/work orders/handoff docs.
- Frontend one-time merge script was removed:
  - `scripts/merge_frontend.sh` (deleted on 2026-02-07)
  - Reason: source tree `frontend_steps/` no longer exists and script had destructive `rm -rf frontend`.

## Executable Shell Scripts

| Script | Reference Count* | Status | Action |
|---|---:|---|---|
| `run_all_pipeline.sh` | 5 | Active | Keep (canonical full run wrapper) |
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
| `scripts/check_gold_data.py` | 0 | Manual utility | Keep (review later) |
| `scripts/convert_md_to_html.py` | 0 | Manual utility | Keep (review later) |
| `scripts/generate_final_report.py` | 0 | Manual utility | Keep (review later) |
| `scripts/visualize_agents.py` | 0 | Manual utility | Keep (review later) |

\* Reference count was measured with repository text search excluding archive-heavy paths (`archive/**`, `docs/archive/**`) and excluding self-file matches.

## Next Cleanup Candidates

1. Decide whether `scripts/setup_scheduler.sh` should move to `docs/manuals/` example-only snippet.
2. Review 0-reference Python utilities and either document exact runbook usage or archive them.
