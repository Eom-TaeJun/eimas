# EIMAS Refactor Handoff

Last updated: 2026-02-06
Owner context: EIMAS bloat-reduction and structure redesign for `main.py --full`.

## 1. Goal and Working Policy

- Keep behavioral baseline of full pipeline (`python main.py --full`).
- Prioritize structure decomposition and contract clarity over repeated full end-to-end runs.
- Split work into:
  - Advanced reasoning/design/contract validation lane.
  - General code move/edit/delete lane (parallelizable).

## 2. Major Changes Already Applied

### 2.1 Analyzer split (monolith -> shards)

- `pipeline/analyzers.py` is now a facade/re-export module.
- New shard modules:
  - `pipeline/analyzers_core.py`
  - `pipeline/analyzers_advanced.py`
  - `pipeline/analyzers_quant.py`
  - `pipeline/analyzers_sentiment.py`
  - `pipeline/analyzers_governance.py`
- `pipeline/analyzers_extension.py` converted to deprecated compatibility shim.

### 2.2 Main orchestration and execution contract fixes

- `main.py` now supports compatibility params:
  - `output_dir`
  - `cron_mode`
- Output path handling unified across Phase 5/6/7/8 save points:
  - JSON/MD artifacts are now written consistently to the same output path.
- Relative `output_dir` is interpreted project-root relative.
- CLI options added:
  - `--output-dir`
  - `--cron-mode`

### 2.6 Phase migration progress (M2 expanded)

- Logic moved out of `main.py` into `pipeline/phases/*`:
  - `pipeline/phases/phase1_collect.py`
  - `pipeline/phases/phase2_basic.py`
  - `pipeline/phases/phase2_enhanced.py`
  - `pipeline/phases/phase2_adjustment.py`
  - `pipeline/phases/phase3_debate.py`
  - `pipeline/phases/phase4_realtime.py`
  - `pipeline/phases/phase45_operational.py`
  - `pipeline/phases/phase5_storage.py`
  - `pipeline/phases/phase6_portfolio.py`
  - `pipeline/phases/phase7_report.py`
  - `pipeline/phases/phase8_validation.py`
- `main.py` line count reduced to ~`341` lines and now behaves as orchestration-first entrypoint.
- `main.py` binds phase handlers via aliases/wrappers in one section.

### 2.7 Artifact path contract hardening (new)

- Added single-run artifact path policy:
  - new ADR: `docs/architecture/ADV_007_ARTIFACT_PATH_POLICY_V1.md`
- `pipeline/storage.py`:
  - `save_result_json(...)` now supports optional `output_file`.
  - when provided, later phases update the same JSON file instead of creating new timestamp files.
- `main.py` now threads the Phase 5 `output_file` into:
  - `phase7_report.generate_report(...)`
  - `phase7_report.validate_report(...)`
  - `phase8_validation.run_ai_validation_phase(...)`

### 2.8 Data schema robustness hardening (new)

- `pipeline/phases/phase6_portfolio.py`:
  - backtest price extraction now handles `Close`/`close`/`Adj Close` variants.
- `pipeline/phases/phase2_enhanced.py`:
  - strategic allocation SPY return extraction now uses the same close-column fallback strategy.

### 2.3 Runner compatibility fix

- `pipeline/runner.py` wrapper now forwards `output_dir` and `cron_mode` to canonical `main.run_integrated_pipeline(...)`.
- Previously these arguments were ignored.

### 2.4 Execution contract script hardening

- `scripts/check_execution_contract.sh` added/refined:
  - project-root detection based on script path (portable, no hardcoded absolute path).
  - explicit status tokens (`STATUS:PASS`, `STATUS:FAIL`) to avoid false grep matches.
  - local/external backend source checks + allocation/rebalancing smoke test.

### 2.5 Utility script portability

- `run_all_pipeline.sh` now `cd`s based on script location instead of hardcoded absolute path.

### 2.9 Entrypoint cleanup (new)

- Removed `main_integrated.py` compatibility shim.
- Canonical execution entrypoint is now only:
  - `main.py` / `run_integrated_pipeline(...)`
- Removed `api/server.py` (legacy API entrypoint); canonical API entry is:
  - `api/main.py`
- Removed legacy runtime trees:
  - `lib/deprecated/`
  - `archive/legacy/`

## 3. Validation Status

Validated in this workspace:

- `python3 -m py_compile main.py pipeline/runner.py cli/eimas.py pipeline/analyzers_extension.py`
- `python3 -m py_compile pipeline/phases/__init__.py pipeline/phases/common.py pipeline/phases/phase1_collect.py pipeline/phases/phase2_basic.py pipeline/phases/phase2_enhanced.py pipeline/phases/phase2_adjustment.py pipeline/phases/phase3_debate.py pipeline/phases/phase4_realtime.py pipeline/phases/phase45_operational.py pipeline/phases/phase5_storage.py pipeline/phases/phase6_portfolio.py pipeline/phases/phase7_report.py pipeline/phases/phase8_validation.py`
- `python3 -m py_compile pipeline/storage.py`
- `bash -n scripts/check_execution_contract.sh`
- `bash scripts/check_execution_contract.sh` -> PASS (3/3)
- shim alias check:
  - `from pipeline.analyzers_extension import analyze_volume_anomalies`
  - confirms alias maps to facade export.

Not validated here:

- `pytest` suite (`pytest` command missing in current env).
- full E2E run after this latest M2-expanded phase migration.

## 4. Current Repository State (Important)

Working tree is dirty and contains many unrelated or parallel-lane changes.
Do not assume only one feature branch scope.

Observed high-signal changed/new areas include:

- `main.py`, `pipeline/runner.py`, `pipeline/analyzers*.py`
- `pipeline/phases/*` (M2 expanded: real logic now present)
- `scripts/check_execution_contract.sh`
- `docs/architecture/*`
- `work_orders/*`
- frontend and API files also changed in parallel.

## 5. Open Risks / Gaps

- Phase-level runtime tests are still mostly compile/smoke; deeper behavioral regression tests are thin.
- `phase6_portfolio.py` contains complex DB/backtest side effects and should get focused test coverage.
- `save_result_md(...)` still creates timestamp-based new file each call (JSON path policy only applied).
- Active runtime backup files (`*_backup_*`, `.backup_*`) removed from live paths.
- Legacy runtime trees removed:
  - `lib/deprecated/`
  - `archive/legacy/`
- Remaining cleanup target is historical docs/scripts under `archive/docs` and `docs/session_artifacts`.
- Large parallel change set increases merge/regression risk without stricter contract checks.

## 6. Recommended Next Steps (for next AI)

1. Contract preservation:
   - Keep `run_integrated_pipeline(...)` signature and behavior compatibility.
   - Add focused smoke checks for any moved phase function.
2. Backup/archive cleanup:
   - Continue cleanup for `archive/docs` + `docs/session_artifacts` (keep only audit-meaningful artifacts).
3. Final integration pass:
   - Run one full pipeline execution after structural migration stabilizes.
4. Testing hardening:
   - Add phase-level unit/smoke tests for `phase2_enhanced.py`, `phase6_portfolio.py`, `phase45_operational.py`.

## 7. Fast Commands for Continuation

```bash
cd /home/tj/projects/autoai/eimas
git status --short
python3 -m py_compile main.py pipeline/runner.py pipeline/analyzers.py pipeline/analyzers_core.py pipeline/analyzers_advanced.py pipeline/analyzers_quant.py pipeline/analyzers_sentiment.py pipeline/analyzers_governance.py
bash scripts/check_execution_contract.sh
```

## 8. Architecture Reset Reference

- New ADR: `docs/architecture/ADV_006_RESET_ARCHITECTURE_V1.md`
- Purpose: allow boundary-first redesign (not constrained by legacy file layout).

## 9. EOD Snapshot (2026-02-06)

- Canonical entrypoints unified:
  - pipeline: `main.py`
  - API: `api/main.py`
- Removed bloat groups:
  - active backup files (`*_backup_*`, `.backup_*`)
  - `main_integrated.py`
  - `api/server.py`
  - `lib/deprecated/`
  - `archive/legacy/`
- Root docs cleaned:
  - session/result markdown moved to `docs/archive/root_legacy/`
  - root markdown count reduced (`31 -> 17`)

## 10. Morning Restart Checklist

1. `cd /home/tj/projects/autoai/eimas && git status --short`
2. `python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py`
3. `bash scripts/check_execution_contract.sh`
4. Start Wave-N:
   - clean stale references in `docs/architecture/*` (`main_orchestrator`, `run_full_analysis`)
   - decide retention policy for `archive/docs` and prune low-value files
