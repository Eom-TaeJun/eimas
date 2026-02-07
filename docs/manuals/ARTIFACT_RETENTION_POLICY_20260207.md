# Artifact Retention Policy (2026-02-07)

## Scope

- `archive/docs/`
- `docs/session_artifacts/`

## Retention Principles

1. Keep technical/audit artifacts that explain architecture, implementation decisions, or verification history.
2. Remove personal portfolio/self-introduction documents that are not needed for runtime, testing, or architecture traceability.
3. Remove obsolete one-off task logs when a newer stabilization log or integrated plan already exists.
4. Prefer conservative pruning in small batches and record each deletion set.

## Keep Buckets

- Architecture/implementation evidence:
  - phase completion reports
  - integration reports
  - gap analysis / not-completed trackers
  - project structure summaries
- Stabilization/session records tied to current refactor:
  - `TASK_LOG_STABILIZATION.md`
  - `WALKTHROUGH_STABILIZATION.md`
  - `IMPLEMENTATION_PLAN_REORG_v2.3.0.md`

## 1st Pruning Batch (Applied: 2026-02-07)

### Removed from `archive/docs/`

- `CREDIT_ASSESSMENT_PORTFOLIO.md`
- `SELF_INTRODUCTION_CREDIT.md`
- `SELF_INTRODUCTION_CREDIT_v2.md`
- `PROJECT_INTRODUCTION_FINAL.md`

### Removed from `docs/session_artifacts/`

- `IMPLEMENTATION_PLAN.md`
- `TASK_LOG.md`

## Inventory Delta

- `archive/docs`: `19 -> 15` files
- `docs/session_artifacts`: `6 -> 4` files

## Next Pass Candidates

1. Review Korean narrative-only summary docs in `archive/docs` for merge-or-drop.
2. Keep only one high-signal project summary if multiple files overlap in content.
