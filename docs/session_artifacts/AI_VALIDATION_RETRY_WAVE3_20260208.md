# AI Validation Retry Wave 3 (2026-02-08)

## Scope
- Target: `lib/validation_agents.py` (`ValidationAgentManager`)
- Goal: agent-specific retry policy + selective retry by failure type

## Implemented
- Added `AgentRetryPolicy` and default retryable failure set.
- Added per-agent run trace telemetry (`attempts`, `retries`, `backoff_seconds`, `last_failure_type`, `failure_history`).
- Added failure-type classifier:
  - `timeout`
  - `rate_limit`
  - `transient_network`
  - `server_overload`
  - `auth`
  - `bad_request`
  - `unknown`
- `validate_all(...)` keeps thread fan-out and now calls retry wrapper with agent name.
- `validate_all(...)` aggregates runtime telemetry into `consensus.validation_runtime_stats`.
- `pipeline/analyzers_governance.py` now includes `validation_runtime_stats` in returned `validation_result`.
- Per-agent differentiated defaults:
  - `Perplexity`: `max_retries = default + 1`, `base_backoff_sec = default * 1.5`
  - `Claude/Gemini/GPT`: use default policy
- Optional override env:
  - `EIMAS_VALIDATION_RETRY_POLICY_OVERRIDES` (JSON)

## Override Example
```bash
export EIMAS_VALIDATION_RETRY_POLICY_OVERRIDES='{
  "default": {
    "max_retries": 1,
    "base_backoff_sec": 1.0,
    "retry_on": ["timeout", "rate_limit", "transient_network", "server_overload"]
  },
  "Perplexity": {
    "max_retries": 3,
    "base_backoff_sec": 2.0
  },
  "GPT": {
    "retry_on": ["timeout", "rate_limit"]
  }
}'
```

## Smoke
- `python3 -m py_compile lib/validation_agents.py` -> PASS
- fake-agent selective retry:
  - `429 rate limit` case: Perplexity recovered after retries (`calls: 3`)
  - `API key not configured` case: GPT no retry (`calls: 1`)
- trace check:
  - `total_retries`: `1`
  - `retried_agents`: `['Perplexity']`
  - `failure_type_counts`: `{'auth': 1, 'rate_limit': 1}`
- `bash scripts/check_execution_contract.sh` -> PASS (3/3)

## Notes
- Non-transient failures (`auth`, `bad_request`) are intentionally not retried by default.
- Next optimization candidate: collect live full-run timing budget by phase (`full` baseline on unrestricted network).
